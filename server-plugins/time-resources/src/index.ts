//
// Copyright © 2023 Hardcore Engineering Inc.
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

import contact, { Employee, Person, PersonAccount } from '@hcengineering/contact'
import core, {
  AttachedData,
  Class,
  Data,
  Doc,
  DocumentUpdate,
  Ref,
  SortingOrder,
  Status,
  Tx,
  TxCUD,
  TxCreateDoc,
  TxFactory,
  TxProcessor,
  TxUpdateDoc,
  toIdMap
} from '@hcengineering/core'
import notification, { CommonInboxNotification } from '@hcengineering/notification'
import { getResource } from '@hcengineering/platform'
import type { TriggerControl } from '@hcengineering/server-core'
import {
  getCommonNotificationTxes,
  getNotificationContent,
  isShouldNotifyTx
} from '@hcengineering/server-notification-resources'
import task, { makeRank } from '@hcengineering/task'
import tracker, { Issue, IssueStatus, Project, TimeSpendReport } from '@hcengineering/tracker'
import serverTime, { OnToDo, ToDoFactory } from '@hcengineering/server-time'
import time, { ProjectToDo, ToDo, ToDoPriority, TodoAutomationHelper, WorkSlot } from '@hcengineering/time'

/**
 * @public
 */
export async function OnTask (tx: Tx, control: TriggerControl): Promise<Tx[]> {
  const actualTx = TxProcessor.extractTx(tx) as TxCUD<Doc>
  const mixin = control.hierarchy.classHierarchyMixin<Class<Doc>, ToDoFactory>(
    actualTx.objectClass,
    serverTime.mixin.ToDoFactory
  )
  if (mixin !== undefined) {
    if (actualTx._class !== core.class.TxRemoveDoc) {
      const factory = await getResource(mixin.factory)
      return await factory(tx, control)
    } else {
      const todos = await control.findAll(time.class.ToDo, { attachedTo: actualTx.objectId })
      return todos.map((p) => control.txFactory.createTxRemoveDoc(p._class, p.space, p._id))
    }
  }

  return []
}

export async function OnWorkSlotCreate (tx: Tx, control: TriggerControl): Promise<Tx[]> {
  const actualTx = TxProcessor.extractTx(tx) as TxCUD<WorkSlot>
  if (!control.hierarchy.isDerived(actualTx.objectClass, time.class.WorkSlot)) return []
  if (!control.hierarchy.isDerived(actualTx._class, core.class.TxCreateDoc)) return []
  const workslot = TxProcessor.createDoc2Doc(actualTx as TxCreateDoc<WorkSlot>)
  const workslots = await control.findAll(time.class.WorkSlot, { attachedTo: workslot.attachedTo })
  if (workslots.length > 1) return []
  const todo = (await control.findAll(time.class.ToDo, { _id: workslot.attachedTo }))[0]
  if (todo === undefined) return []
  if (!control.hierarchy.isDerived(todo.attachedToClass, tracker.class.Issue)) return []
  const issue = (await control.findAll(tracker.class.Issue, { _id: todo.attachedTo as Ref<Issue> }))[0]
  if (issue === undefined) return []
  const project = (await control.findAll(task.class.Project, { _id: issue.space }))[0]
  if (project !== undefined) {
    const type = (await control.modelDb.findAll(task.class.ProjectType, { _id: project.type }))[0]
    if (type?.classic) {
      const taskType = (await control.modelDb.findAll(task.class.TaskType, { _id: issue.kind }))[0]
      if (taskType !== undefined) {
        const statuses = await control.findAll(core.class.Status, { _id: { $in: taskType.statuses } })
        const statusMap = toIdMap(statuses)
        const typeStatuses = taskType.statuses.map((p) => statusMap.get(p)).filter((p) => p !== undefined) as Status[]
        const current = statusMap.get(issue.status)
        if (current === undefined) return []
        if (current.category !== task.statusCategory.UnStarted && current.category !== task.statusCategory.ToDo) {
          return []
        }
        const nextStatus = typeStatuses.find((p) => p.category === task.statusCategory.Active)
        if (nextStatus !== undefined) {
          const factory = new TxFactory(control.txFactory.account)
          const innerTx = factory.createTxUpdateDoc(issue._class, issue.space, issue._id, {
            status: nextStatus._id
          })
          const outerTx = factory.createTxCollectionCUD(
            issue.attachedToClass,
            issue.attachedTo,
            issue.space,
            issue.collection,
            innerTx
          )
          await control.apply([outerTx], true)
          return []
        }
      }
    }
  }
  return []
}

export async function OnToDoRemove (tx: Tx, control: TriggerControl): Promise<Tx[]> {
  const actualTx = TxProcessor.extractTx(tx) as TxCUD<ToDo>
  if (!control.hierarchy.isDerived(actualTx.objectClass, time.class.ToDo)) return []
  if (!control.hierarchy.isDerived(actualTx._class, core.class.TxRemoveDoc)) return []
  const todo = control.removedMap.get(actualTx.objectId) as ToDo
  if (todo === undefined) return []
  // it was closed, do nothing
  if (todo.doneOn != null) return []
  const todos = await control.findAll(time.class.ToDo, { attachedTo: todo.attachedTo })
  if (todos.length > 0) return []
  const issue = (await control.findAll(tracker.class.Issue, { _id: todo.attachedTo as Ref<Issue> }))[0]
  if (issue === undefined) return []
  const project = (await control.findAll(task.class.Project, { _id: issue.space }))[0]
  if (project !== undefined) {
    const type = (await control.modelDb.findAll(task.class.ProjectType, { _id: project.type }))[0]
    if (type !== undefined && type.classic) {
      const factory = new TxFactory(control.txFactory.account)
      const taskType = (await control.modelDb.findAll(task.class.TaskType, { _id: issue.kind }))[0]
      if (taskType !== undefined) {
        const statuses = await control.findAll(core.class.Status, { _id: { $in: taskType.statuses } })
        const statusMap = toIdMap(statuses)
        const typeStatuses = taskType.statuses.map((p) => statusMap.get(p)).filter((p) => p !== undefined) as Status[]
        const current = statusMap.get(issue.status)
        if (current === undefined) return []
        if (current.category !== task.statusCategory.Active && current.category !== task.statusCategory.ToDo) return []
        const nextStatus = typeStatuses.find((p) => p.category === task.statusCategory.UnStarted)
        if (nextStatus !== undefined) {
          const innerTx = factory.createTxUpdateDoc(issue._class, issue.space, issue._id, {
            status: nextStatus._id
          })
          const outerTx = factory.createTxCollectionCUD(
            issue.attachedToClass,
            issue.attachedTo,
            issue.space,
            issue.collection,
            innerTx
          )
          await control.apply([outerTx], true)
          return []
        }
      }
    }
  }
  return []
}

export async function OnToDoCreate (tx: TxCUD<Doc>, control: TriggerControl): Promise<Tx[]> {
  const hierarchy = control.hierarchy
  const createTx = TxProcessor.extractTx(tx) as TxCreateDoc<ToDo>

  if (!hierarchy.isDerived(createTx.objectClass, time.class.ToDo)) return []
  if (!hierarchy.isDerived(createTx._class, core.class.TxCreateDoc)) return []

  const mixin = hierarchy.classHierarchyMixin(
    createTx.objectClass as Ref<Class<Doc>>,
    notification.mixin.ClassCollaborators
  )

  if (mixin === undefined) {
    return []
  }

  const todo = TxProcessor.createDoc2Doc(createTx)
  const account = await getPersonAccount(todo.user, control)

  if (account === undefined) {
    return []
  }

  const object = (await control.findAll(todo.attachedToClass, { _id: todo.attachedTo }))[0]

  if (object === undefined) return []

  const res: Tx[] = []
  const notifyResult = await isShouldNotifyTx(control, createTx, tx, todo, account._id, true, false)
  const content = await getNotificationContent(tx, account._id, todo, control)
  const data: Partial<Data<CommonInboxNotification>> = {
    ...content,
    header: time.string.ToDo,
    headerIcon: time.icon.Planned,
    headerObjectId: object._id,
    headerObjectClass: object._class,
    messageHtml: todo.title
  }

  res.push(
    ...(await getCommonNotificationTxes(
      control,
      object,
      data,
      account._id,
      tx.modifiedBy,
      object._id,
      object._class,
      object.space,
      createTx.modifiedOn,
      notifyResult
    ))
  )

  return res
}

/**
 * @public
 */
export async function OnToDoUpdate (tx: Tx, control: TriggerControl): Promise<Tx[]> {
  const actualTx = TxProcessor.extractTx(tx) as TxCUD<ToDo>
  if (!control.hierarchy.isDerived(actualTx.objectClass, time.class.ToDo)) return []
  if (!control.hierarchy.isDerived(actualTx._class, core.class.TxUpdateDoc)) return []
  const updTx = actualTx as TxUpdateDoc<ToDo>
  const doneOn = updTx.operations.doneOn
  const title = updTx.operations.title
  const description = updTx.operations.description
  const visibility = updTx.operations.visibility
  if (doneOn != null) {
    const events = await control.findAll(time.class.WorkSlot, { attachedTo: updTx.objectId })
    const res: Tx[] = []
    const resEvents: WorkSlot[] = []
    for (const event of events) {
      if (event.date > doneOn) {
        const innerTx = control.txFactory.createTxRemoveDoc(event._class, event.space, event._id)
        const outerTx = control.txFactory.createTxCollectionCUD(
          event.attachedToClass,
          event.attachedTo,
          event.space,
          event.collection,
          innerTx
        )
        res.push(outerTx)
      } else if (event.dueDate > doneOn) {
        const upd: DocumentUpdate<WorkSlot> = {
          dueDate: doneOn
        }
        if (title !== undefined) {
          upd.title = title
        }
        if (description !== undefined) {
          upd.description = description
        }
        const innerTx = control.txFactory.createTxUpdateDoc(event._class, event.space, event._id, upd)
        const outerTx = control.txFactory.createTxCollectionCUD(
          event.attachedToClass,
          event.attachedTo,
          event.space,
          event.collection,
          innerTx
        )
        res.push(outerTx)
        resEvents.push({
          ...event,
          dueDate: doneOn
        })
      } else {
        resEvents.push(event)
      }
    }
    const todo = (await control.findAll(time.class.ToDo, { _id: updTx.objectId }))[0]
    if (todo === undefined) return res
    const funcs = control.hierarchy.classHierarchyMixin<Class<Doc>, OnToDo>(
      todo.attachedToClass,
      serverTime.mixin.OnToDo
    )
    if (funcs !== undefined) {
      const func = await getResource(funcs.onDone)
      const todoRes = await func(control, resEvents, todo)
      await control.apply(todoRes, true)
    }
    return res
  }
  if (title !== undefined || description !== undefined || visibility !== undefined) {
    const events = await control.findAll(time.class.WorkSlot, { attachedTo: updTx.objectId })
    const res: Tx[] = []
    for (const event of events) {
      const upd: DocumentUpdate<WorkSlot> = {}
      if (title !== undefined) {
        upd.title = title
      }
      if (description !== undefined) {
        upd.description = description
      }
      if (visibility !== undefined) {
        const newVisibility = visibility === 'public' ? 'public' : 'freeBusy'
        if (event.visibility !== newVisibility) {
          upd.visibility = newVisibility
        }
      }
      const innerTx = control.txFactory.createTxUpdateDoc(event._class, event.space, event._id, upd)
      const outerTx = control.txFactory.createTxCollectionCUD(
        event.attachedToClass,
        event.attachedTo,
        event.space,
        event.collection,
        innerTx
      )
      res.push(outerTx)
    }
    return res
  }
  return []
}

/**
 * @public
 */
export async function IssueToDoFactory (tx: Tx, control: TriggerControl): Promise<Tx[]> {
  const actualTx = TxProcessor.extractTx(tx) as TxCUD<Issue>
  if (!control.hierarchy.isDerived(actualTx.objectClass, tracker.class.Issue)) return []
  if (control.hierarchy.isDerived(actualTx._class, core.class.TxCreateDoc)) {
    const issue = TxProcessor.createDoc2Doc(actualTx as TxCreateDoc<Issue>)
    return await createIssueHandler(issue, control)
  } else if (control.hierarchy.isDerived(actualTx._class, core.class.TxUpdateDoc)) {
    const updateTx = actualTx as TxUpdateDoc<Issue>
    return await updateIssueHandler(updateTx, control)
  }
  return []
}

/**
 * @public
 */
export async function IssueToDoDone (control: TriggerControl, workslots: WorkSlot[], todo: ToDo): Promise<Tx[]> {
  const res: Tx[] = []
  let total = 0
  for (const workslot of workslots) {
    total += (workslot.dueDate - workslot.date) / 1000 / 60
  }
  const factory = new TxFactory(control.txFactory.account)
  const issue = (await control.findAll<Issue>(todo.attachedToClass, { _id: todo.attachedTo as Ref<Issue> }))[0]
  if (issue !== undefined) {
    const project = (await control.findAll(task.class.Project, { _id: issue.space }))[0]
    if (project !== undefined) {
      const type = (await control.modelDb.findAll(task.class.ProjectType, { _id: project.type }))[0]
      if (type?.classic) {
        const taskType = (await control.modelDb.findAll(task.class.TaskType, { _id: issue.kind }))[0]
        if (taskType !== undefined) {
          const index = taskType.statuses.findIndex((p) => p === issue.status)

          const helpers = await control.modelDb.findAll<TodoAutomationHelper>(time.class.TodoAutomationHelper, {})
          const testers = await Promise.all(helpers.map((it) => getResource(it.onDoneTester)))
          let allowed = true
          for (const tester of testers) {
            if (!(await tester(control, todo))) {
              allowed = false
              break
            }
          }
          if (index !== -1 && allowed) {
            const nextStatus = taskType.statuses[index + 1]
            if (nextStatus !== undefined) {
              const currentStatus = taskType.statuses[index]
              const current = (await control.findAll(core.class.Status, { _id: currentStatus }))[0]
              const next = (await control.findAll(core.class.Status, { _id: nextStatus }))[0]
              if (
                current.category !== task.statusCategory.Lost &&
                next.category !== task.statusCategory.Lost &&
                current.category !== task.statusCategory.Won
              ) {
                const innerTx = factory.createTxUpdateDoc(issue._class, issue.space, issue._id, {
                  status: nextStatus
                })
                const outerTx = factory.createTxCollectionCUD(
                  issue.attachedToClass,
                  issue.attachedTo,
                  issue.space,
                  issue.collection,
                  innerTx
                )
                res.push(outerTx)
              }
            }
          }
        }
      }
    }

    if (total > 0) {
      // round to nearest 15 minutes
      total = Math.round(total / 15) * 15

      const data: AttachedData<TimeSpendReport> = {
        employee: todo.user as Ref<Employee>,
        date: new Date().getTime(),
        value: total / 60,
        description: ''
      }
      const innerTx = factory.createTxCreateDoc(
        tracker.class.TimeSpendReport,
        issue.space,
        data as Data<TimeSpendReport>
      )
      const outerTx = factory.createTxCollectionCUD(issue._class, issue._id, issue.space, 'reports', innerTx)
      res.push(outerTx)
    }
  }
  return res
}

async function createIssueHandler (issue: Issue, control: TriggerControl): Promise<Tx[]> {
  if (issue.assignee != null) {
    const project = (await control.findAll(task.class.Project, { _id: issue.space }))[0]
    if (project === undefined) return []
    const type = (await control.modelDb.findAll(task.class.ProjectType, { _id: project.type }))[0]
    if (!type?.classic) return []
    const status = (await control.findAll(core.class.Status, { _id: issue.status }))[0]
    if (status === undefined) return []
    if (status.category === task.statusCategory.Active || status.category === task.statusCategory.ToDo) {
      const tx = await getCreateToDoTx(issue, issue.assignee, control)
      if (tx !== undefined) {
        await control.apply([tx], true)
      }
    }
  }
  return []
}

async function getPersonAccount (person: Ref<Person>, control: TriggerControl): Promise<PersonAccount | undefined> {
  const account = (
    await control.modelDb.findAll(
      contact.class.PersonAccount,
      {
        person
      },
      { limit: 1 }
    )
  )[0]
  return account
}

async function getIssueToDoData (
  issue: Issue,
  user: Ref<Person>,
  control: TriggerControl
): Promise<AttachedData<ProjectToDo> | undefined> {
  const acc = await getPersonAccount(user, control)
  if (acc === undefined) return
  const firstTodoItem = (
    await control.findAll(
      time.class.ToDo,
      {
        user: acc.person,
        doneOn: null
      },
      {
        limit: 1,
        sort: { rank: SortingOrder.Ascending }
      }
    )
  )[0]
  const rank = makeRank(undefined, firstTodoItem?.rank)
  const data: AttachedData<ProjectToDo> = {
    attachedSpace: issue.space,
    workslots: 0,
    description: '',
    priority: ToDoPriority.NoPriority,
    visibility: 'public',
    title: issue.title,
    user: acc.person,
    rank
  }
  return data
}

async function getCreateToDoTx (issue: Issue, user: Ref<Person>, control: TriggerControl): Promise<Tx | undefined> {
  const data = await getIssueToDoData(issue, user, control)
  if (data === undefined) return
  const innerTx = control.txFactory.createTxCreateDoc(
    time.class.ProjectToDo,
    time.space.ToDos,
    data as Data<ProjectToDo>
  )
  innerTx.space = core.space.Tx
  const outerTx = control.txFactory.createTxCollectionCUD(issue._class, issue._id, time.space.ToDos, 'todos', innerTx)
  outerTx.space = core.space.Tx
  return outerTx
}

async function changeIssueAssigneeHandler (
  control: TriggerControl,
  newAssignee: Ref<Person>,
  issueId: Ref<Issue>
): Promise<Tx[]> {
  const issue = (await control.findAll(tracker.class.Issue, { _id: issueId }))[0]
  if (issue !== undefined) {
    const status = (await control.findAll(core.class.Status, { _id: issue.status }))[0]
    if (status === undefined) return []
    if (status.category === task.statusCategory.Active) {
      const tx = await getCreateToDoTx(issue, newAssignee, control)
      if (tx !== undefined) return [tx]
    }
  }
  return []
}

async function changeIssueStatusHandler (
  control: TriggerControl,
  newStatus: Ref<IssueStatus>,
  issueId: Ref<Issue>
): Promise<Tx[]> {
  const status = (await control.findAll(core.class.Status, { _id: newStatus }))[0]
  if (status === undefined) return []
  if (status.category === task.statusCategory.Active || status.category === task.statusCategory.ToDo) {
    const issue = (await control.findAll(tracker.class.Issue, { _id: issueId }))[0]
    if (issue?.assignee != null) {
      const todos = await control.findAll(time.class.ToDo, {
        attachedTo: issue._id,
        user: issue.assignee
      })
      if (todos.length === 0) {
        const tx = await getCreateToDoTx(issue, issue.assignee, control)
        if (tx !== undefined) {
          await control.apply([tx], true)
        }
      }
    }
  }
  return []
}

async function changeIssueNumberHandler (control: TriggerControl, issueId: Ref<Issue>): Promise<Tx[]> {
  const res: Tx[] = []
  const issue = (await control.findAll(tracker.class.Issue, { _id: issueId }))[0]
  if (issue !== undefined) {
    const todos = await control.findAll(time.class.ToDo, {
      attachedTo: issue._id
    })
    for (const todo of todos) {
      const data = await getIssueToDoData(issue, todo.user, control)
      if (data === undefined) continue
      const update: DocumentUpdate<ToDo> = {}
      if (data.title !== todo.title) {
        update.title = data.title
      }
      if (data.description !== todo.description) {
        update.description = data.description
      }
      if (data.attachedSpace !== todo.attachedSpace) {
        update.attachedSpace = data.attachedSpace
      }
      if (Object.keys(update).length > 0) {
        const innerTx = control.txFactory.createTxUpdateDoc(todo._class, todo.space, todo._id, update)
        const outerTx = control.txFactory.createTxCollectionCUD(
          issue._class,
          issue._id,
          time.space.ToDos,
          'todos',
          innerTx
        )
        res.push(outerTx)
      }
    }
  }
  return res
}

async function updateIssueHandler (tx: TxUpdateDoc<Issue>, control: TriggerControl): Promise<Tx[]> {
  const res: Tx[] = []
  const project = (await control.findAll(task.class.Project, { _id: tx.objectSpace as Ref<Project> }))[0]
  if (project === undefined) return []
  const type = (await control.modelDb.findAll(task.class.ProjectType, { _id: project.type }))[0]
  if (!type?.classic) return []
  const newAssignee = tx.operations.assignee
  if (newAssignee != null) {
    res.push(...(await changeIssueAssigneeHandler(control, newAssignee, tx.objectId)))
  }
  const newStatus = tx.operations.status
  if (newStatus !== undefined) {
    res.push(...(await changeIssueStatusHandler(control, newStatus, tx.objectId)))
  }
  const number = tx.operations.number
  if (number !== undefined) {
    res.push(...(await changeIssueNumberHandler(control, tx.objectId)))
  }
  return res
}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
export default async () => ({
  function: {
    IssueToDoFactory,
    IssueToDoDone
  },
  trigger: {
    OnTask,
    OnToDoUpdate,
    OnToDoRemove,
    OnToDoCreate,
    OnWorkSlotCreate
  }
})
