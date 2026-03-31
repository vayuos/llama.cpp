import * as vscode from 'vscode';
import * as net from 'net';
import axios from 'axios';
import { ServerLauncher } from './serverLauncher';
import { loadConfig } from '../config/loadConfig';
import { validateConfig } from '../config/validateConfig';

export enum ProcessStatus {
    STOPPED = 'STOPPED',
    STARTING = 'STARTING',
    RUNNING = 'RUNNING',
    ERROR = 'ERROR',
    OFFLINE = 'OFFLINE'
}

export class ProcessManager {
    private coderTerminal: vscode.Terminal | undefined;
    private reviewerTerminal: vscode.Terminal | undefined;
    private statuses = { coder: ProcessStatus.STOPPED, reviewer: ProcessStatus.STOPPED };

    async startAll() {
        const config = loadConfig();
        try {
            validateConfig(config);
        } catch (e: any) {
            vscode.window.showErrorMessage(e.message);
            return;
        }

        const coderPort = parseInt(config.coder.endpoint.split(':')[2]);
        const reviewerPort = parseInt(config.reviewer.endpoint.split(':')[2]);

        if (!(await this.checkPort(coderPort))) {
            vscode.window.showErrorMessage(`Gravitas: Coder port ${coderPort} is already in use.`);
            return;
        }
        if (!(await this.checkPort(reviewerPort))) {
            vscode.window.showErrorMessage(`Gravitas: Reviewer port ${reviewerPort} is already in use.`);
            return;
        }

        this.statuses.coder = ProcessStatus.STARTING;
        this.statuses.reviewer = ProcessStatus.STARTING;

        this.reviewerTerminal = ServerLauncher.launchReviewer(config);
        this.coderTerminal = ServerLauncher.launchCoder(config);

        // Start health polling
        this.pollHealth(config.coder.endpoint, 'coder');
        this.pollHealth(config.reviewer.endpoint, 'reviewer');
    }

    private async checkPort(port: number): Promise<boolean> {
        return new Promise((resolve) => {
            const server = net.createServer();
            server.once('error', () => resolve(false));
            server.once('listening', () => {
                server.close();
                resolve(true);
            });
            server.listen(port);
        });
    }

    private async pollHealth(endpoint: string, type: 'coder' | 'reviewer') {
        const interval = setInterval(async () => {
            if (this.statuses[type] === ProcessStatus.STOPPED) {
                clearInterval(interval);
                return;
            }

            try {
                await axios.get(`${endpoint}/v1/models`, { timeout: 2000 });
                this.statuses[type] = ProcessStatus.RUNNING;
            } catch (e) {
                // If the terminal still exists, it's starting or failed
                const term = type === 'coder' ? this.coderTerminal : this.reviewerTerminal;
                if (term) {
                    this.statuses[type] = ProcessStatus.STARTING;
                } else {
                    this.statuses[type] = ProcessStatus.OFFLINE;
                    clearInterval(interval);
                }
            }
        }, 5000);
    }

    stopAll() {
        if (this.coderTerminal) {
            this.coderTerminal.dispose();
            this.coderTerminal = undefined;
        }
        if (this.reviewerTerminal) {
            this.reviewerTerminal.dispose();
            this.reviewerTerminal = undefined;
        }
        this.statuses.coder = ProcessStatus.STOPPED;
        this.statuses.reviewer = ProcessStatus.STOPPED;
    }

    restartAll() {
        this.stopAll();
        setTimeout(() => this.startAll(), 1000);
    }

    handleTerminalClosed(terminal: vscode.Terminal) {
        if (terminal === this.coderTerminal) {
            this.coderTerminal = undefined;
            this.statuses.coder = ProcessStatus.STOPPED;
        } else if (terminal === this.reviewerTerminal) {
            this.reviewerTerminal = undefined;
            this.statuses.reviewer = ProcessStatus.STOPPED;
        }
    }

    getStatus(): { coder: ProcessStatus, reviewer: ProcessStatus } {
        return this.statuses;
    }
}
