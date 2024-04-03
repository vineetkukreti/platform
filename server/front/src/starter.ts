//
// Copyright © 2020, 2021 Anticrm Platform Contributors.
// Copyright © 2021 Hardcore Engineering Inc.
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

import { MeasureContext } from '@hcengineering/core'
import { setMetadata } from '@hcengineering/platform'
import { buildStorageFromConfig, storageConfigFromEnv } from '@hcengineering/server'
import { StorageConfiguration } from '@hcengineering/server-core'
import serverToken from '@hcengineering/server-token'
import { start } from '.'

export function startFront (ctx: MeasureContext, extraConfig?: Record<string, string>): void {
  const defaultLanguage = process.env.DEFAULT_LANGUAGE ?? 'en'
  const languages = process.env.LANGUAGES ?? 'en,ru'
  const SERVER_PORT = parseInt(process.env.SERVER_PORT ?? '8080')

  const transactorEndpoint = process.env.TRANSACTOR_URL
  if (transactorEndpoint === undefined) {
    console.error('please provide transactor url')
    process.exit(1)
  }

  const url = process.env.MONGO_URL
  if (url === undefined) {
    console.error('please provide mongodb url')
    process.exit(1)
  }

  const elasticUrl = process.env.ELASTIC_URL
  if (elasticUrl === undefined) {
    console.error('please provide elastic url')
    process.exit(1)
  }

  const storageConfig: StorageConfiguration = storageConfigFromEnv()
  const storageAdapter = buildStorageFromConfig(storageConfig, url)

  const accountsUrl = process.env.ACCOUNTS_URL
  if (accountsUrl === undefined) {
    console.error('please provide accounts url')
    process.exit(1)
  }

  const uploadUrl = process.env.UPLOAD_URL
  if (uploadUrl === undefined) {
    console.error('please provide upload url')
    process.exit(1)
  }

  const gmailUrl = process.env.GMAIL_URL
  if (gmailUrl === undefined) {
    console.error('please provide gmail url')
    process.exit(1)
  }

  const calendarUrl = process.env.CALENDAR_URL
  if (calendarUrl === undefined) {
    console.error('please provide calendar service url')
    process.exit(1)
  }

  const telegramUrl = process.env.TELEGRAM_URL
  if (telegramUrl === undefined) {
    console.error('please provide telegram url')
    process.exit(1)
  }

  const rekoniUrl = process.env.REKONI_URL
  if (rekoniUrl === undefined) {
    console.error('please provide rekoni url')
    process.exit(1)
  }

  const collaboratorUrl = process.env.COLLABORATOR_URL
  if (collaboratorUrl === undefined) {
    console.error('please provide collaborator url')
    process.exit(1)
  }

  const collaboratorApiUrl = process.env.COLLABORATOR_API_URL
  if (collaboratorApiUrl === undefined) {
    console.error('please provide collaborator api url')
    process.exit(1)
  }

  const modelVersion = process.env.MODEL_VERSION
  if (modelVersion === undefined) {
    console.error('please provide model version requirement')
    process.exit(1)
  }

  const serverSecret = process.env.SERVER_SECRET
  if (serverSecret === undefined) {
    console.log('Please provide server secret')
    process.exit(1)
  }

  const lastNameFirst = process.env.LAST_NAME_FIRST

  const title = process.env.TITLE

  setMetadata(serverToken.metadata.Secret, serverSecret)

  const config = {
    transactorEndpoint,
    elasticUrl,
    storageAdapter,
    accountsUrl,
    uploadUrl,
    modelVersion,
    gmailUrl,
    lastNameFirst,
    telegramUrl,
    rekoniUrl,
    calendarUrl,
    collaboratorUrl,
    collaboratorApiUrl,
    title,
    languages,
    defaultLanguage
  }
  console.log('Starting Front service with', config)
  const shutdown = start(ctx, config, SERVER_PORT, extraConfig)

  const close = (): void => {
    console.trace('Exiting from server')
    console.log('Shutdown request accepted')
    shutdown()
    process.exit(0)
  }

  process.on('SIGINT', close)
  process.on('SIGTERM', close)
}
