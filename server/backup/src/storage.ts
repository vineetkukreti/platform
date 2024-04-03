import { MeasureContext, WorkspaceId } from '@hcengineering/core'
import { StorageAdapter } from '@hcengineering/server-core'
import { createReadStream, createWriteStream, existsSync } from 'fs'
import { mkdir, readFile, writeFile } from 'fs/promises'
import { dirname, join } from 'path'
import { PassThrough, Readable, Writable } from 'stream'

/**
 * @public
 */
export interface BackupStorage {
  loadFile: (name: string) => Promise<Buffer>
  load: (name: string) => Promise<Readable>
  write: (name: string) => Promise<Writable>
  writeFile: (name: string, data: string | Buffer) => Promise<void>
  exists: (name: string) => Promise<boolean>
}

class FileStorage implements BackupStorage {
  constructor (readonly root: string) {}
  async loadFile (name: string): Promise<Buffer> {
    return await readFile(join(this.root, name))
  }

  async write (name: string): Promise<Writable> {
    const fileName = join(this.root, name)
    const dir = dirname(fileName)
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true })
    }

    return createWriteStream(join(this.root, name))
  }

  async load (name: string): Promise<Readable> {
    return createReadStream(join(this.root, name))
  }

  async exists (name: string): Promise<boolean> {
    return existsSync(join(this.root, name))
  }

  async writeFile (name: string, data: string | Buffer): Promise<void> {
    const fileName = join(this.root, name)
    const dir = dirname(fileName)
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true })
    }

    await writeFile(fileName, data)
  }
}

class AdapterStorage implements BackupStorage {
  constructor (
    readonly client: StorageAdapter,
    readonly workspaceId: WorkspaceId,
    readonly root: string,
    readonly ctx: MeasureContext
  ) {}

  async loadFile (name: string): Promise<Buffer> {
    const data = await this.client.read(this.ctx, this.workspaceId, join(this.root, name))
    return Buffer.concat(data)
  }

  async write (name: string): Promise<Writable> {
    const wr = new PassThrough()
    void this.client.put(this.ctx, this.workspaceId, join(this.root, name), wr, 'application/octet-stream')
    return wr
  }

  async load (name: string): Promise<Readable> {
    return await this.client.get(this.ctx, this.workspaceId, join(this.root, name))
  }

  async exists (name: string): Promise<boolean> {
    try {
      return (await this.client.stat(this.ctx, this.workspaceId, join(this.root, name))) !== undefined
    } catch (err) {
      return false
    }
  }

  async writeFile (name: string, data: string | Buffer): Promise<void> {
    // TODO: add mime type detection here.
    await this.client.put(
      this.ctx,
      this.workspaceId,
      join(this.root, name),
      data,
      'application/octet-stream',
      data.length
    )
  }
}

/**
 * @public
 */
export async function createFileBackupStorage (fileName: string): Promise<BackupStorage> {
  if (!existsSync(fileName)) {
    await mkdir(fileName, { recursive: true })
  }
  return new FileStorage(fileName)
}

/**
 * @public
 */
export async function createStorageBackupStorage (
  ctx: MeasureContext,
  client: StorageAdapter,
  workspaceId: WorkspaceId,
  root: string
): Promise<BackupStorage> {
  if (!(await client.exists(ctx, workspaceId))) {
    await client.make(ctx, workspaceId)
  }
  return new AdapterStorage(client, workspaceId, root, ctx)
}
