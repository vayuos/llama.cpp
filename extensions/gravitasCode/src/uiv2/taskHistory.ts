import * as vscode from 'vscode';
import { TaskManager } from './taskManager';
import { Task } from './types';

/**
 * Provides a tree view of historical Tasks.
 * (Gap 5: Task History)
 */
export class TaskHistoryProvider implements vscode.TreeDataProvider<TaskTreeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<TaskTreeItem | undefined | null | void> = new vscode.EventEmitter<TaskTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<TaskTreeItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor() {
        TaskManager.getInstance().onDidTaskUpdate(() => this.refresh());
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: TaskTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: TaskTreeItem): Promise<TaskTreeItem[]> {
        if (element) return [];

        const tasks = TaskManager.getInstance().getAllTasks();
        return tasks.map(task => new TaskTreeItem(task));
    }
}

class TaskTreeItem extends vscode.TreeItem {
    constructor(public readonly task: Task) {
        super(task.command, vscode.TreeItemCollapsibleState.None);
        
        this.tooltip = `${task.status}: ${task.command}`;
        this.description = new Date(task.createdAt).toLocaleTimeString();
        this.contextValue = 'task';
        
        // Icon based on status
        this.iconPath = this._getIcon(task.status);

        this.command = {
            command: 'gravitas.task.openInShell',
            title: 'Open in chat',
            arguments: [task.id]
        };
    }

    private _getIcon(status: string) {
        switch (status) {
            case 'COMPLETED': return new vscode.ThemeIcon('check', new vscode.ThemeColor('debugIcon.startForeground'));
            case 'FAILED': return new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));
            case 'RUNNING': return new vscode.ThemeIcon('sync~spin');
            case 'ABORTED': return new vscode.ThemeIcon('stop');
            default: return new vscode.ThemeIcon('circle-outline');
        }
    }
}
