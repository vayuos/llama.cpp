"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.TaskHistoryProvider = void 0;
const vscode = __importStar(require("vscode"));
const taskManager_1 = require("./taskManager");
/**
 * Provides a tree view of historical Tasks.
 * (Gap 5: Task History)
 */
class TaskHistoryProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        taskManager_1.TaskManager.getInstance().onDidTaskUpdate(() => this.refresh());
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    async getChildren(element) {
        const logger = require('../core/logger').CentralLogger.getInstance();
        if (element)
            return [];
        const tasks = taskManager_1.TaskManager.getInstance().getAllTasks();
        logger.debug('system', `TaskHistoryProvider: Fetched ${tasks.length} tasks from TaskManager.`);
        return tasks.map(task => new TaskTreeItem(task));
    }
}
exports.TaskHistoryProvider = TaskHistoryProvider;
class TaskTreeItem extends vscode.TreeItem {
    constructor(task) {
        super(task.command, vscode.TreeItemCollapsibleState.None);
        this.task = task;
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
    _getIcon(status) {
        switch (status) {
            case 'COMPLETED': return new vscode.ThemeIcon('check', new vscode.ThemeColor('debugIcon.startForeground'));
            case 'FAILED': return new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));
            case 'RUNNING': return new vscode.ThemeIcon('sync~spin');
            case 'ABORTED': return new vscode.ThemeIcon('stop');
            default: return new vscode.ThemeIcon('circle-outline');
        }
    }
}
//# sourceMappingURL=taskHistory.js.map