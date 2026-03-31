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
exports.ContextManager = void 0;
const cp = __importStar(require("child_process"));
const vscode = __importStar(require("vscode"));
const astParser_1 = require("./astParser");
class ContextManager {
    static async init() {
        await this.astParser.init();
    }
    static async findSymbols(query) {
        return new Promise((resolve, reject) => {
            const root = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
            if (!root)
                return resolve([]);
            cp.exec(`rg -l "${query}" ${root}`, (err, stdout) => {
                if (err && err.code !== 1)
                    return reject(err);
                if (!stdout)
                    return resolve([]);
                const files = stdout.split('\n').filter(f => f.trim() !== '');
                resolve(files.slice(0, 5));
            });
        });
    }
    static async getDefinition(file, symbol) {
        const content = await this.getFileContent(file);
        const boundaries = await this.astParser.getFunctionBoundaries(content);
        // Find boundary matching symbol
        return new Promise((resolve, reject) => {
            cp.exec(`ctags -f - --excmd=number ${file} | grep "${symbol}"`, (err, stdout) => {
                if (err)
                    return resolve('');
                resolve(stdout.trim());
            });
        });
    }
    static async getFileContent(file) {
        const doc = await vscode.workspace.openTextDocument(file);
        return doc.getText();
    }
}
exports.ContextManager = ContextManager;
ContextManager.astParser = new astParser_1.ASTParser();
//# sourceMappingURL=contextManager.js.map