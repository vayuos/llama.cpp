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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const module_1 = __importDefault(require("module"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
// Mock VS Code
const mockVscode = {
    EventEmitter: class {
        constructor() {
            this.event = (handler) => { this.handler = handler; };
        }
        fire(data) { if (this.handler)
            this.handler(data); }
    },
    Disposable: class {
    },
    Uri: { file: (p) => ({ fsPath: p }) }
};
// Hook module resolution
const originalResolve = module_1.default._resolveFilename;
module_1.default._resolveFilename = function (request, parent, isMain) {
    if (request === 'vscode')
        return 'vscode';
    return originalResolve.apply(this, arguments);
};
module_1.default._cache['vscode'] = {
    id: 'vscode',
    filename: 'vscode',
    loaded: true,
    exports: mockVscode
};
global.vscode = mockVscode;
const taskManager_1 = require("../uiv2/taskManager");
async function main() {
    console.log('🚀 Starting Gravitas Hardening Test Suite');
    const testSpace = path.join(__dirname, 'gravitas_test_space');
    if (!fs.existsSync(testSpace))
        fs.mkdirSync(testSpace);
    if (!fs.existsSync(path.join(testSpace, 'global')))
        fs.mkdirSync(path.join(testSpace, 'global'));
    const mockContext = {
        storageUri: { fsPath: testSpace },
        globalStorageUri: { fsPath: path.join(testSpace, 'global') },
        globalState: {
            update: (key, val) => Promise.resolve(),
            get: (key) => undefined
        },
        workspaceState: {
            update: (key, val) => Promise.resolve(),
            get: (key) => undefined
        },
        subscriptions: [],
        extensionPath: __dirname,
        asAbsolutePath: (p) => path.join(__dirname, p)
    };
    const taskManager = taskManager_1.TaskManager.initialize(mockContext);
    // 1. Telemetry & Resource Guard Test
    await (async () => {
        console.log('\n--- Running Telemetry & Resource Guard Test ---');
        const task = await taskManager.createTask('Test Telemetry', 'user');
        taskManager.startNextAttempt(task.id);
        taskManager.startPhase(task.id, 'coder', 'Testing Telemetry');
        const telemetryListener = taskManager.onDidEmitEvent(({ event }) => {
            if (event.type === 'ResourceUsageSampled') {
                const r = event.resources;
                console.log(`📊 Sample: RAM=${r.ramMb}MB, CPU=${r.cpuPercent}%, VRAM=${r.vramMb}MB`);
            }
            else if (event.type === 'ResourceLimitExceeded') {
                console.log(`⚠️ LIMIT EXCEEDED: ${event.resourceType} ${event.actualValue} (Limit ${event.limitValue}) Severity=${event.severity}`);
            }
        });
        console.log('Baseline sample...');
        taskManager.sampleResources(task.id);
        console.log('Simulating CPU Load (Fibonacci)...');
        const start = Date.now();
        function fib(n) {
            if (n <= 1)
                return n;
            return fib(n - 1) + fib(n - 2);
        }
        while (Date.now() - start < 1000) {
            fib(30);
        }
        console.log('Post-load sample...');
        taskManager.sampleResources(task.id);
    })();
    // 2. Artifact Validation Pipeline Test
    await (async () => {
        console.log('\n--- Running Artifact Validation Pipeline Test ---');
        const task = await taskManager.createTask('Test Artifacts', 'user');
        taskManager.startNextAttempt(task.id);
        taskManager.startPhase(task.id, 'coder', 'Generating File');
        taskManager.onDidEmitEvent(({ event }) => {
            if (event.type === 'ArtifactProduced') {
                console.log('📦 ArtifactProduced: ' + event.path);
            }
            else if (event.type === 'ArtifactValidated') {
                console.log(`✅ ArtifactValidated: Status=${event.status} | Validator=${event.validatorId} | Message: ${event.message}`);
            }
        });
        const dummyFile = path.join(testSpace, `test_artifact_${Date.now()}.txt`);
        fs.writeFileSync(dummyFile, 'Hello Gravitas Execution Ledger');
        console.log('Recording valid artifact...');
        taskManager.recordArtifact(task.id, dummyFile, 'text/plain');
        console.log('Recording non-existent artifact (fail test)...');
        taskManager.recordArtifact(task.id, path.join(testSpace, 'missing.txt'), 'text/plain');
        if (fs.existsSync(dummyFile))
            fs.unlinkSync(dummyFile);
    })();
    console.log('\nCleaning up test workspace...');
    // Small delay to ensure all async file writes (like event logging) are complete
    await new Promise(resolve => setTimeout(resolve, 500));
    fs.rmSync(testSpace, { recursive: true, force: true });
    console.log('🏁 All Tests Complete');
}
main().catch(error => {
    console.error('❌ Test Suite Failed:', error);
    process.exit(1);
});
//# sourceMappingURL=telemetryTest.js.map