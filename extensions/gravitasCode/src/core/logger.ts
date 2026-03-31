import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

export type LogSource = 'validation' | 'coder' | 'reviewer' | 'ui' | 'system';
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
    timestamp: string;
    source: LogSource;
    message: string;
    level: LogLevel;
}

const LEVEL_WEIGHTS: Record<LogLevel, number> = {
    'debug': 0,
    'info': 1,
    'warn': 2,
    'error': 3
};

/**
 * Centralized Logging Engine for Gravitas.
 * Supports granular levels and dynamic filtering based on user settings.
 */
export class CentralLogger {
    private static instance: CentralLogger;
    private outputChannel: vscode.OutputChannel;
    private logFile: string | null = null;
    private minLevelWeight: number = 0; // Default to debug until initialized
    private _onDidLog = new vscode.EventEmitter<LogEntry>();
    public readonly onDidLog = this._onDidLog.event;

    private constructor() {
        this.outputChannel = vscode.window.createOutputChannel('Gravitas Logs');
        // Initial setup for early boot logs
        this.setLogDir(path.join(os.homedir(), '.gravitas', 'logs'));
    }

    public static getInstance(): CentralLogger {
        if (!CentralLogger.instance) {
            CentralLogger.instance = new CentralLogger();
        }
        return CentralLogger.instance;
    }

    /**
     * Sets the minimum log level to display.
     */
    public setLevel(level: LogLevel) {
        this.minLevelWeight = LEVEL_WEIGHTS[level] ?? 0;
    }

    public setLogDir(logDir: string) {
        if (!fs.existsSync(logDir)) {
            try {
                fs.mkdirSync(logDir, { recursive: true });
            } catch (e) {
                console.error('Failed to create log dir:', e);
                return;
            }
        }
        this.logFile = path.join(logDir, `gravitas-${new Date().toISOString().split('T')[0]}.log`);
    }

    public log(source: LogSource, message: string, level: LogLevel = 'info') {
        const weight = LEVEL_WEIGHTS[level] ?? 1;
        
        // 🛡️ Filtering: Only output if level meets minimum priority
        if (weight < this.minLevelWeight) {
            return;
        }

        const entry: LogEntry = {
            timestamp: new Date().toISOString(),
            source,
            message,
            level
        };

        const formatted = `[${entry.timestamp}] [${entry.source.toUpperCase()}] [${entry.level.toUpperCase()}] ${entry.message}`;
        this.outputChannel.appendLine(formatted);

        if (this.logFile) {
            try {
                fs.appendFileSync(this.logFile, formatted + '\n');
            } catch (e) {}
        }

        this._onDidLog.fire(entry);
    }

    public debug(source: LogSource, message: string) { this.log(source, message, 'debug'); }
    public info(source: LogSource, message: string) { this.log(source, message, 'info'); }
    public warn(source: LogSource, message: string) { this.log(source, message, 'warn'); }
    public error(source: LogSource, message: string) { this.log(source, message, 'error'); }
}
