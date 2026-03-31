import Module from 'module';
import * as fs from 'fs';
import * as path from 'path';

// Mock VS Code
const mockVscode: any = {
    EventEmitter: class {
        event = (handler: any) => { this.handler = handler; };
        handler: any;
        fire(data: any) { if (this.handler) this.handler(data); }
    },
    Disposable: class { },
    Uri: { file: (p: string) => ({ fsPath: p }) }
};

// Hook module resolution
const originalResolve = (Module as any)._resolveFilename;
(Module as any)._resolveFilename = function (request: string, parent: any, isMain: boolean) {
    if (request === 'vscode') return 'vscode';
    return originalResolve.apply(this, arguments);
};

(Module as any)._cache['vscode'] = {
    id: 'vscode',
    filename: 'vscode',
    loaded: true,
    exports: mockVscode
};

(global as any).vscode = mockVscode;

import { TaskManager } from '../uiv2/taskManager';

async function main() {
    console.log('🚀 Starting Gravitas Hardening Test Suite');

    const testSpace = path.join(__dirname, 'gravitas_test_space');
    if (!fs.existsSync(testSpace)) fs.mkdirSync(testSpace);
    if (!fs.existsSync(path.join(testSpace, 'global'))) fs.mkdirSync(path.join(testSpace, 'global'));

    const mockContext: any = {
        storageUri: { fsPath: testSpace },
        globalStorageUri: { fsPath: path.join(testSpace, 'global') },
        globalState: {
            update: (key: string, val: any) => Promise.resolve(),
            get: (key: string) => undefined
        },
        workspaceState: {
            update: (key: string, val: any) => Promise.resolve(),
            get: (key: string) => undefined
        },
        subscriptions: [],
        extensionPath: __dirname,
        asAbsolutePath: (p: string) => path.join(__dirname, p)
    };

    const taskManager = (TaskManager as any).initialize(mockContext);

    // 1. Telemetry & Resource Guard Test
    await (async () => {
        console.log('\n--- Running Telemetry & Resource Guard Test ---');
        const task = await taskManager.createTask('Test Telemetry', 'user');

        taskManager.startNextAttempt(task.id);
        taskManager.startPhase(task.id, 'coder', 'Testing Telemetry');

        const telemetryListener = taskManager.onDidEmitEvent(({ event }: any) => {
            if (event.type === 'ResourceUsageSampled') {
                const r = event.resources;
                console.log(`📊 Sample: RAM=${r.ramMb}MB, CPU=${r.cpuPercent}%, VRAM=${r.vramMb}MB`);
            } else if (event.type === 'ResourceLimitExceeded') {
                console.log(`⚠️ LIMIT EXCEEDED: ${event.resourceType} ${event.actualValue} (Limit ${event.limitValue}) Severity=${event.severity}`);
            }
        });

        console.log('Baseline sample...');
        taskManager.sampleResources(task.id);

        console.log('Simulating CPU Load (Fibonacci)...');
        const start = Date.now();
        function fib(n: number): number {
            if (n <= 1) return n;
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

        taskManager.onDidEmitEvent(({ event }: any) => {
            if (event.type === 'ArtifactProduced') {
                console.log('📦 ArtifactProduced: ' + event.path);
            } else if (event.type === 'ArtifactValidated') {
                console.log(`✅ ArtifactValidated: Status=${event.status} | Validator=${event.validatorId} | Message: ${event.message}`);
            }
        });

        const dummyFile = path.join(testSpace, `test_artifact_${Date.now()}.txt`);
        fs.writeFileSync(dummyFile, 'Hello Gravitas Execution Ledger');

        console.log('Recording valid artifact...');
        taskManager.recordArtifact(task.id, dummyFile, 'text/plain');

        console.log('Recording non-existent artifact (fail test)...');
        taskManager.recordArtifact(task.id, path.join(testSpace, 'missing.txt'), 'text/plain');

        if (fs.existsSync(dummyFile)) fs.unlinkSync(dummyFile);
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
