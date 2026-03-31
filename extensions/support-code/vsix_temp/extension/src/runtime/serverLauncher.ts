import * as vscode from 'vscode';
import { GravitasConfig } from '../config/gravitasConfig';

export class ServerLauncher {
    static launchCoder(config: GravitasConfig): vscode.Terminal {
        const name = 'Gravitas: Coder';
        let terminal = vscode.window.terminals.find(t => t.name === name);
        if (terminal) terminal.dispose();
        terminal = vscode.window.createTerminal(name);

        const c = config.coder;
        const cmd = [
            `CUDA_VISIBLE_DEVICES=${c.cudaDevices}`,
            c.binaryPath,
            `-m ${c.modelPath}`,
            `--host ${c.endpoint.split(':')[1].replace('//', '')}`,
            `--port ${c.endpoint.split(':')[2]}`,
            `-ngl ${c.gpuLayers}`,
            `-c ${c.contextSize}`,
            `--threads ${c.threads}`
        ].join(' ');

        terminal.show();
        terminal.sendText(cmd);
        return terminal;
    }

    static launchReviewer(config: GravitasConfig): vscode.Terminal {
        const name = 'Gravitas: Reviewer';
        let terminal = vscode.window.terminals.find(t => t.name === name);
        if (terminal) terminal.dispose();
        terminal = vscode.window.createTerminal(name);

        const r = config.reviewer;
        const cmd = [
            r.binaryPath,
            `-m ${r.modelPath}`,
            `--host ${r.endpoint.split(':')[1].replace('//', '')}`,
            `--port ${r.endpoint.split(':')[2]}`,
            `-ngl 0`,
            `-c 8192`,
            `--threads ${r.threads}`,
            `--temp 0.0`
        ].join(' ');

        terminal.show();
        terminal.sendText(cmd);
        return terminal;
    }
}
