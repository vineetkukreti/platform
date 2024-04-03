//
// Copyright © 2020, 2021 Anticrm Platform Contributors.
// Copyright © 2022 Hardcore Engineering Inc.
//
// Licensed under the Eclipse Public License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License. You may
// obtain a copy of the License at https://www.eclipse.org/legal/epl-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//
// See the License for the specific language governing permissions and
// limitations under the License.
//

import core, {
  type AttachedDoc,
  type Class,
  type Doc,
  type DocIndexState,
  type DocumentQuery,
  type FindOptions,
  type FindResult,
  type Hierarchy,
  type MeasureContext,
  type ObjQueryType,
  type Ref,
  type SearchOptions,
  type SearchQuery,
  type SearchResult,
  type ServerStorage,
  type Tx,
  type TxCUD,
  type TxCollectionCUD,
  TxFactory,
  type TxResult,
  type WorkspaceId,
  docKey,
  isFullTextAttribute,
  isIndexedAttribute,
  toFindResult
} from '@hcengineering/core'
import { type FullTextIndexPipeline } from './indexer'
import { createStateDoc, isClassIndexable } from './indexer/utils'
import { getScoringConfig, mapSearchResultDoc } from './mapper'
import { type StorageAdapter } from './storage'
import type { FullTextAdapter, IndexedDoc, WithFind } from './types'

/**
 * @public
 */
export class FullTextIndex implements WithFind {
  txFactory = new TxFactory(core.account.System, true)
  consistency: Promise<void> | undefined

  constructor (
    private readonly hierarchy: Hierarchy,
    private readonly adapter: FullTextAdapter,
    private readonly dbStorage: ServerStorage,
    readonly storageAdapter: StorageAdapter | undefined,
    readonly workspace: WorkspaceId,
    readonly indexer: FullTextIndexPipeline,
    private readonly upgrade: boolean
  ) {
    if (!upgrade) {
      // Schedule indexing after consistency check
      void this.indexer.startIndexing()
    }
  }

  async close (): Promise<void> {
    await this.indexer.cancel()
    await this.consistency
  }

  async tx (ctx: MeasureContext, txes: Tx[]): Promise<TxResult> {
    const stDocs = new Map<Ref<DocIndexState>, { create?: DocIndexState, updated: boolean, removed: boolean }>()
    for (let tx of txes) {
      let attachedTo: Ref<DocIndexState> | undefined
      let attachedToClass: Ref<Class<Doc>> | undefined
      if (tx._class === core.class.TxCollectionCUD) {
        const txcol = tx as TxCollectionCUD<Doc, AttachedDoc>
        attachedTo = txcol.objectId as Ref<DocIndexState>
        attachedToClass = txcol.objectClass
        tx = txcol.tx
      }
      if (this.hierarchy.isDerived(tx._class, core.class.TxCUD)) {
        const cud = tx as TxCUD<Doc>

        if (!isClassIndexable(this.hierarchy, cud.objectClass)) {
          // No need, since no indixable fields or attachments.
          continue
        }

        let stDoc: DocIndexState | undefined
        if (cud._class === core.class.TxCreateDoc) {
          // Add doc for indexing
          stDoc = createStateDoc(cud.objectId, cud.objectClass, {
            attributes: {},
            stages: {},
            attachedTo,
            attachedToClass,
            space: tx.objectSpace,
            removed: false
          })
          stDocs.set(cud.objectId as Ref<DocIndexState>, { create: stDoc, updated: false, removed: false })
        } else {
          const old = stDocs.get(cud.objectId as Ref<DocIndexState>)
          if (cud._class === core.class.TxRemoveDoc && old?.create !== undefined) {
            // Object created and deleted, skip index
            stDocs.delete(cud.objectId as Ref<DocIndexState>)
            continue
          } else {
            // Create and update
            if (old?.removed === true) continue
            else {
              stDocs.set(cud.objectId as Ref<DocIndexState>, {
                ...old,
                create: cud._class !== core.class.TxRemoveDoc ? old?.create : undefined,
                updated: cud._class !== core.class.TxRemoveDoc && old?.create === undefined,
                removed: cud._class === core.class.TxRemoveDoc
              })
            }
          }
        }
      }
    }
    await ctx.with('queue', {}, async (ctx) => {
      await this.indexer.queue(ctx, stDocs)
    })
    this.indexer.triggerIndexing()
    return {}
  }

  async findAll<T extends Doc>(
    ctx: MeasureContext,
    _class: Ref<Class<T>>,
    query: DocumentQuery<T>,
    options?: FindOptions<T>
  ): Promise<FindResult<T>> {
    const { _id, $search, ...mainQuery } = query
    if ($search === undefined) return toFindResult([])

    const ids: Set<Ref<Doc>> = new Set<Ref<Doc>>()
    const baseClass = this.hierarchy.getBaseClass(_class)
    let classes = this.hierarchy.getDescendants(baseClass)

    const attrs = this.hierarchy.getAllAttributes(_class)

    // We need to filter all non indexed fields from query to make it work properly
    const findQuery: DocumentQuery<Doc> = {
      $search: query.$search
    }
    try {
      for (const [k, attr] of attrs) {
        if (isFullTextAttribute(attr) || isIndexedAttribute(attr)) {
          const vv = (query as any)[k]
          if (vv != null) {
            if (
              k === '_class' ||
              k === 'modifiedBy' ||
              k === 'modifiedOn' ||
              k === 'space' ||
              k === 'attachedTo' ||
              k === 'attachedToClass'
            ) {
              findQuery[k] = vv
            } else {
              const docKeyValue = docKey(attr.name, { _class: attr.attributeOf })
              findQuery[docKeyValue] = vv
            }
          }
        }
        if (attr.type._class === core.class.Collection) {
          // we need attached documents to be in classes
          const dsc = this.hierarchy.getDescendants(attr.attributeOf)
          classes = classes.concat(dsc)
        }
      }
    } catch (err: any) {
      console.error(err)
    }

    classes = classes.filter((it, idx, arr) => arr.indexOf(it) === idx)

    classes = classes.filter((it) => {
      if (typeof query._class === 'object') {
        if (query._class?.$in !== undefined) {
          return query._class.$in.includes(it)
        }
        if (query._class?.$nin !== undefined) {
          return !query._class.$nin.includes(it)
        }
      }
      return true
    })

    const fullTextLimit = Math.min(5000, (options?.limit ?? 200) * 100)

    let { docs, pass } = await this.indexer.search(classes, findQuery, fullTextLimit)

    if (docs.length === 0 && pass) {
      docs = await this.adapter.search(classes, findQuery, fullTextLimit)
    }
    const indexedDocMap = new Map<Ref<Doc>, IndexedDoc>()

    for (const doc of docs) {
      if (doc._class.some((cl) => this.hierarchy.isDerived(cl, baseClass))) {
        ids.add(doc.id)
        indexedDocMap.set(doc.id, doc)
      }
      if (doc.attachedTo != null) {
        if (doc.attachedToClass != null && this.hierarchy.isDerived(doc.attachedToClass, baseClass)) {
          if (this.hierarchy.isDerived(doc.attachedToClass, baseClass)) {
            ids.add(doc.attachedTo)
            indexedDocMap.set(doc.attachedTo, doc)
          }
        } else {
          ids.add(doc.attachedTo)
          indexedDocMap.set(doc.attachedTo, doc)
        }
      }
    }
    if (docs.length === 0) {
      return toFindResult([], 0)
    }
    const scoreSearch: number | undefined = (options?.sort as any)?.['#score']

    const resultIds = Array.from(getResultIds(ids, _id))
    let result = await this.dbStorage.findAll(
      ctx,
      _class,
      { _id: { $in: resultIds }, ...mainQuery },
      {
        ...options,
        limit: scoreSearch !== undefined ? docs.length : options?.limit
      }
    )

    // Just assign scores based on idex
    result.forEach((it) => {
      const idDoc = indexedDocMap.get(it._id)
      const { _score, id, _class, ...extra } = idDoc as any
      it.$source = {
        ...extra,
        $score: _score
      }
    })
    if (scoreSearch !== undefined) {
      result.sort((a, b) => scoreSearch * ((a.$source?.$score ?? 0) - (b.$source?.$score ?? 0)))
    }
    if (scoreSearch !== undefined) {
      if (options?.limit !== undefined && options?.limit < result.length) {
        result = toFindResult(result.slice(0, options?.limit), result.total)
      }
    }
    return result
  }

  async searchFulltext (ctx: MeasureContext, query: SearchQuery, options: SearchOptions): Promise<SearchResult> {
    const resultRaw = await this.adapter.searchString(query, {
      ...options,
      scoring: getScoringConfig(this.hierarchy, query.classes ?? [])
    })

    const result: SearchResult = {
      ...resultRaw,
      docs: resultRaw.docs.map((raw) => {
        return mapSearchResultDoc(this.hierarchy, raw)
      })
    }
    return result
  }

  submitting: Promise<void> | undefined

  timeout: any
}

function getResultIds (ids: Set<Ref<Doc>>, _id: ObjQueryType<Ref<Doc>> | undefined): Set<Ref<Doc>> {
  const result = new Set<Ref<Doc>>()
  if (_id !== undefined) {
    if (typeof _id === 'string') {
      if (!ids.has(_id)) {
        return new Set()
      } else {
        result.add(_id)
      }
    } else if (_id.$in !== undefined) {
      for (const id of _id.$in) {
        if (ids.has(id)) {
          result.add(id)
        }
      }
    } else if (_id.$nin !== undefined) {
      for (const id of ids) {
        if (!_id.$nin.includes(id)) {
          result.add(id)
        }
      }
    }
  } else {
    return ids
  }
  return result
}
