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
    private outputChannel: vscode.OutputChannel | null = null;
    private logFile: string | null = null;
    private minLevelWeight: number = 0; // Default to debug until initialized
    private _onDidLog = new vscode.EventEmitter<LogEntry>();
    public readonly onDidLog = this._onDidLog.event;
    private isLogging: boolean = false;
    private isEnabled: boolean = false; // 🛡️ Safe Boot: Events disabled until activate() completes

    private constructor() {
        // 🛡️ Zero-API Constructor: No VS Code calls here to avoid early boot crashes.
        // Initial setup for early boot logs (pure path logic)
        this.setLogDir(path.join(os.homedir(), '.gravitas', 'logs'));
    }

    private getOutputChannel(): vscode.OutputChannel {
        if (!this.outputChannel) {
            this.outputChannel = vscode.window.createOutputChannel('Gravitas Logs');
        }
        return this.outputChannel;
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

    /**
     * Safety: Enables firing of onDidLog events. Should only be called
     * at the end of the activation sequence.
     */
    public enableEvents() {
        this.isEnabled = true;
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
        
        // 🛡️ Loop Prevention: Immediate return if we are already in a logging call
        if (this.isLogging) {
            return;
        }

        // 🛡️ Filtering: Only output if level meets minimum priority
        if (weight < this.minLevelWeight) {
            return;
        }

        this.isLogging = true;
        try {
            const entry: LogEntry = {
                timestamp: new Date().toISOString(),
                source,
                message,
                level
            };

            const formatted = `[${entry.timestamp}] [${entry.source.toUpperCase()}] [${entry.level.toUpperCase()}] ${entry.message}`;
            this.getOutputChannel().appendLine(formatted);

            if (this.logFile) {
                try {
                    fs.appendFileSync(this.logFile, formatted + '\n');
                } catch (e) {}
            }

            if (this.isEnabled) {
                this._onDidLog.fire(entry);
            }
        } finally {
            this.isLogging = false;
        }
    }

    public debug(source: LogSource, message: string) { this.log(source, message, 'debug'); }
    public info(source: LogSource, message: string) { this.log(source, message, 'info'); }
    public warn(source: LogSource, message: string) { this.log(source, message, 'warn'); }
    public error(source: LogSource, message: string) { this.log(source, message, 'error'); }
}
