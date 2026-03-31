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
exports.PresetsTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
const presetDefinitions_1 = require("../core/presetDefinitions");
class PresetsTreeProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        vscode.commands.registerCommand('gravitas.presets.apply', async (presetKey) => {
            const preset = presetDefinitions_1.GRAVITAS_PRESETS[presetKey];
            if (!preset) {
                vscode.window.showErrorMessage(`Unknown preset: ${presetKey}`);
                return;
            }
            const config = vscode.workspace.getConfiguration('gravitasCode');
            // Apply each setting
            for (const [key, value] of Object.entries(preset.config)) {
                // Split key (e.g., 'coder.mode') to find sub-sections if needed, 
                // but usually workspace.update works with dot notation if section is root.
                // However, 'gravitasCode' is the root, so we pass the relative key.
                await config.update(key, value, vscode.ConfigurationTarget.Global);
            }
            vscode.window.showInformationMessage(`Applied preset: ${preset.label}`);
        });
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element)
            return Promise.resolve([]);
        const items = Object.entries(presetDefinitions_1.GRAVITAS_PRESETS).map(([key, preset]) => {
            return new PresetItem(preset.label, preset.description, key);
        });
        return Promise.resolve(items);
    }
}
exports.PresetsTreeProvider = PresetsTreeProvider;
class PresetItem extends vscode.TreeItem {
    constructor(label, description, presetKey) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.label = label;
        this.description = description;
        this.presetKey = presetKey;
        this.tooltip = `${label}: ${description}`;
        this.iconPath = new vscode.ThemeIcon('package');
        this.command = {
            command: 'gravitas.presets.apply',
            title: 'Apply Preset',
            arguments: [this.presetKey]
        };
    }
}
//# sourceMappingURL=presetsTreeProvider.js.map