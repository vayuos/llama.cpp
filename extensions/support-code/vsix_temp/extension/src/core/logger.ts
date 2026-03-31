import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export type LogSource = 'validation' | 'coder' | 'reviewer' | 'ui' | 'system';

export interface LogEntry {
    timestamp: string;
    source: LogSource;
    message: string;
    level: 'info' | 'warn' | 'error';
}

export class CentralLogger {
    private static instance: CentralLogger;
    private outputChannel: vscode.OutputChannel;
    private logFile: string | null = null;
    private _onDidLog = new vscode.EventEmitter<LogEntry>();
    public readonly onDidLog = this._onDidLog.event;

    private constructor() {
        this.outputChannel = vscode.window.createOutputChannel('Gravitas Logs');
    }

    public static getInstance(): CentralLogger {
        if (!CentralLogger.instance) {
            CentralLogger.instance = new CentralLogger();
        }
        return CentralLogger.instance;
    }

    public setLogDir(logDir: string) {
        if (!fs.existsSync(logDir)) {
            fs.mkdirSync(logDir, { recursive: true });
        }
        this.logFile = path.join(logDir, `gravitas-${new Date().toISOString().split('T')[0]}.log`);
    }

    public log(source: LogSource, message: string, level: 'info' | 'warn' | 'error' = 'info') {
        const entry: LogEntry = {
            timestamp: new Date().toISOString(),
            source,
            message,
            level
        };

        const formatted = `[${entry.timestamp}] [${entry.source.toUpperCase()}] [${entry.level.toUpperCase()}] ${entry.message}`;
        this.outputChannel.appendLine(formatted);

        if (this.logFile) {
            fs.appendFileSync(this.logFile, formatted + '\n');
        }

        this._onDidLog.fire(entry);
    }

    public info(source: LogSource, message: string) { this.log(source, message, 'info'); }
    public warn(source: LogSource, message: string) { this.log(source, message, 'warn'); }
    public error(source: LogSource, message: string) { this.log(source, message, 'error'); }
}
