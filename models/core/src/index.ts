//
// Copyright © 2020, 2021 Anticrm Platform Contributors.
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

import {
  AccountRole,
  systemAccountEmail,
  type AttachedDoc,
  type Class,
  type Doc,
  type DocIndexState,
  type IndexingConfiguration,
  type TxCollectionCUD
} from '@hcengineering/core'
import { type Builder } from '@hcengineering/model'
import core from './component'
import {
  TArrOf,
  TAttachedDoc,
  TAttribute,
  TBlob,
  TBlobData,
  TClass,
  TCollection,
  TConfiguration,
  TConfigurationElement,
  TDoc,
  TDocIndexState,
  TEnum,
  TEnumOf,
  TFullTextSearchContext,
  TFulltextData,
  TIndexConfiguration,
  TIndexStageState,
  TInterface,
  TMigrationState,
  TMixin,
  TObj,
  TPluginConfiguration,
  TRefTo,
  TType,
  TTypeAny,
  TTypeAttachment,
  TTypeBoolean,
  TTypeCollaborativeDoc,
  TTypeCollaborativeDocVersion,
  TTypeCollaborativeMarkup,
  TTypeDate,
  TTypeHyperlink,
  TTypeIntlString,
  TTypeMarkup,
  TTypeNumber,
  TTypeRank,
  TTypeRecord,
  TTypeRelatedDocument,
  TTypeString,
  TTypeTimestamp,
  TVersion
} from './core'
import { TAccount, TSpace, TSpaceType, TSpaceTypeDescriptor, TTypedSpace, TRole, TPermission } from './security'
import { TStatus, TStatusCategory } from './status'
import { TUserStatus } from './transient'
import {
  TTx,
  TTxApplyIf,
  TTxCUD,
  TTxCollectionCUD,
  TTxCreateDoc,
  TTxMixin,
  TTxRemoveDoc,
  TTxUpdateDoc,
  TTxWorkspaceEvent
} from './tx'
import { definePermissions } from './permissions'

export { coreId } from '@hcengineering/core'
export * from './core'
export { coreOperation } from './migration'
export * from './security'
export * from './status'
export * from './tx'
export { core as default }

export function createModel (builder: Builder): void {
  builder.createModel(
    TObj,
    TDoc,
    TClass,
    TMixin,
    TInterface,
    TTx,
    TTxCUD,
    TTxCreateDoc,
    TAttachedDoc,
    TTxCollectionCUD,
    TTxMixin,
    TTxUpdateDoc,
    TTxRemoveDoc,
    TTxApplyIf,
    TTxWorkspaceEvent,
    TSpace,
    TTypedSpace,
    TSpaceType,
    TSpaceTypeDescriptor,
    TRole,
    TPermission,
    TAccount,
    TAttribute,
    TType,
    TEnumOf,
    TTypeMarkup,
    TTypeCollaborativeDoc,
    TTypeCollaborativeDocVersion,
    TTypeCollaborativeMarkup,
    TArrOf,
    TRefTo,
    TTypeDate,
    TTypeTimestamp,
    TTypeNumber,
    TTypeBoolean,
    TTypeString,
    TTypeRank,
    TTypeRecord,
    TTypeAttachment,
    TTypeHyperlink,
    TCollection,
    TVersion,
    TTypeIntlString,
    TPluginConfiguration,
    TUserStatus,
    TEnum,
    TTypeAny,
    TBlobData,
    TFulltextData,
    TTypeRelatedDocument,
    TDocIndexState,
    TIndexStageState,
    TFullTextSearchContext,
    TConfiguration,
    TConfigurationElement,
    TIndexConfiguration,
    TStatus,
    TStatusCategory,
    TMigrationState,
    TBlob
  )

  builder.createDoc(
    core.class.Account,
    core.space.Model,
    {
      email: systemAccountEmail,
      role: AccountRole.Owner
    },
    core.account.System
  )

  builder.mixin<Class<TxCollectionCUD<Doc, AttachedDoc>>, IndexingConfiguration<TxCollectionCUD<Doc, AttachedDoc>>>(
    core.class.TxCollectionCUD,
    core.class.Class,
    core.mixin.IndexConfiguration,
    {
      indexes: [
        'tx.objectId',
        'tx._class',
        'tx.objectClass',
        'tx.operations.attachedTo',
        'space',
        'objectSpace',
        {
          _class: 1,
          objectSpace: 1,
          _id: 1,
          modifiedOn: 1
        },
        {
          _class: 1,
          _id: 1,
          modifiedOn: 1
        }
      ]
    }
  )

  builder.mixin<Class<DocIndexState>, IndexingConfiguration<TxCollectionCUD<Doc, AttachedDoc>>>(
    core.class.DocIndexState,
    core.class.Class,
    core.mixin.IndexConfiguration,
    {
      indexes: [
        {
          _class: 1,
          stages: 1,
          _id: 1,
          modifiedOn: 1
        },
        {
          _class: 1,
          _id: 1,
          modifiedOn: 1
        },
        {
          _class: 1,
          _id: 1,
          objectClass: 1
        }
      ]
    }
  )

  builder.mixin(core.class.Space, core.class.Class, core.mixin.FullTextSearchContext, {
    childProcessingAllowed: false
  })

  definePermissions(builder)
}
