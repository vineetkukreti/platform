//
// Copyright © 2021 Anticrm Platform Contributors.
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

import { toWorkspaceString, type WorkspaceId } from '@hcengineering/core'
import { PlatformError, unknownStatus } from '@hcengineering/platform'
import { type Db, MongoClient, type MongoClientOptions } from 'mongodb'

const connections = new Map<string, MongoClientReferenceImpl>()

// Register mongo close on process exit.
process.on('exit', () => {
  shutdown().catch((err) => {
    console.error(err)
  })
})

/**
 * @public
 */
export async function shutdown (): Promise<void> {
  for (const c of connections.values()) {
    c.close(true)
  }
  connections.clear()
}

export interface MongoClientReference {
  getClient: () => Promise<MongoClient>
  close: () => void
}

class MongoClientReferenceImpl {
  count: number
  client: MongoClient | Promise<MongoClient>

  constructor (
    client: MongoClient | Promise<MongoClient>,
    readonly onclose: () => void
  ) {
    this.count = 0
    this.client = client
  }

  async getClient (): Promise<MongoClient> {
    if (this.client instanceof Promise) {
      this.client = await this.client
    }
    return this.client
  }

  close (force: boolean = false): void {
    this.count--
    if (this.count === 0 || force) {
      if (force) {
        this.count = 0
      }
      this.onclose()
      void (async () => {
        await (await this.client).close()
      })()
    }
  }

  addRef (): void {
    this.count++
  }
}

export class ClientRef implements MongoClientReference {
  constructor (readonly client: MongoClientReferenceImpl) {}

  closed = false
  async getClient (): Promise<MongoClient> {
    if (!this.closed) {
      return await this.client.getClient()
    } else {
      throw new PlatformError(unknownStatus('Mongo client is already closed'))
    }
  }

  close (): void {
    // Do not allow double close of mongo connection client
    if (!this.closed) {
      this.closed = true
      this.client.close()
    }
  }
}

/**
 * Initialize a workspace connection to DB
 * @public
 */
export function getMongoClient (uri: string, options?: MongoClientOptions): MongoClientReference {
  const extraOptions = JSON.parse(process.env.MONGO_OPTIONS ?? '{}')
  const key = `${uri}${process.env.MONGO_OPTIONS}_${JSON.stringify(options)}`
  let existing = connections.get(key)

  // If not created or closed
  if (existing === undefined) {
    existing = new MongoClientReferenceImpl(
      MongoClient.connect(uri, {
        ...options,
        enableUtf8Validation: false,
        ...extraOptions
      }),
      () => {
        connections.delete(key)
      }
    )
    connections.set(key, existing)
  }
  // Add reference and return once closable
  existing.addRef()
  return new ClientRef(existing)
}

/**
 * @public
 *
 * Construct MongoDB table from workspace.
 */
export function getWorkspaceDB (client: MongoClient, workspaceId: WorkspaceId): Db {
  return client.db(toWorkspaceString(workspaceId))
}
