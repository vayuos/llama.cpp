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
exports.ContextTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const contextManager_1 = require("../core/contextManager");
class ContextTreeProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.contextManager = contextManager_1.ContextManager.getInstance();
        vscode.commands.registerCommand('gravitas.context.toggle', (item) => {
            this.contextManager.toggleExclusion(item.path);
            this.refresh();
        });
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    async getChildren(element) {
        if (!element) {
            // Root: Workspace Folders
            const workspaceFolders = vscode.workspace.workspaceFolders;
            if (!workspaceFolders)
                return Promise.resolve([]);
            return workspaceFolders.map(folder => new ContextItem(folder.name, folder.uri.fsPath, vscode.TreeItemCollapsibleState.Collapsed, !this.contextManager.isExcluded(folder.uri.fsPath), 0 // Root folders don't show size directly usually, or we could sum it up
            ));
        }
        else {
            // Recursive directory listing
            const directoryPath = element.path;
            return new Promise((resolve) => {
                fs.readdir(directoryPath, { withFileTypes: true }, (err, dirents) => {
                    if (err) {
                        resolve([]);
                        return;
                    }
                    const items = dirents.map(dirent => {
                        const fullPath = path.join(directoryPath, dirent.name);
                        const isExcluded = this.contextManager.isExcluded(fullPath);
                        let collapsibleState = vscode.TreeItemCollapsibleState.None;
                        let tokenCount = 0;
                        if (dirent.isDirectory()) {
                            collapsibleState = vscode.TreeItemCollapsibleState.Collapsed;
                        }
                        else {
                            tokenCount = this.contextManager.getTokenEstimate(fullPath);
                        }
                        return new ContextItem(dirent.name, fullPath, collapsibleState, !isExcluded, tokenCount);
                    });
                    // Sort: Directories first, then files
                    items.sort((a, b) => {
                        if (a.collapsibleState !== b.collapsibleState) {
                            return a.collapsibleState === vscode.TreeItemCollapsibleState.Collapsed ? -1 : 1;
                        }
                        return a.label.localeCompare(b.label);
                    });
                    resolve(items);
                });
            });
        }
    }
}
exports.ContextTreeProvider = ContextTreeProvider;
class ContextItem extends vscode.TreeItem {
    constructor(label, path, collapsibleState, included = true, tokenCount = 0) {
        super(label, collapsibleState);
        this.label = label;
        this.path = path;
        this.collapsibleState = collapsibleState;
        this.included = included;
        this.tokenCount = tokenCount;
        this.iconPath = new vscode.ThemeIcon(included ? 'check' : 'circle-slash');
        // Visual dimming for excluded items
        if (!included) {
            this.resourceUri = vscode.Uri.file(path); // Helps with file icons
            // We can't easily dim without custom color contribs, but the icon helps.
        }
        const formattedTokens = contextManager_1.ContextManager.getInstance().formatTokenCount(tokenCount);
        this.description = included
            ? (collapsibleState === vscode.TreeItemCollapsibleState.None ? `${formattedTokens} tokens` : '')
            : 'Excluded';
        this.command = {
            command: 'gravitas.context.toggle',
            title: 'Toggle Context',
            arguments: [this]
        };
        // Set context value for potential menu contributions
        this.contextValue = included ? 'included' : 'excluded';
    }
}
//# sourceMappingURL=contextTreeProvider.js.map