import * as cp from 'child_process';
import { SymbolResolver } from './symbolResolver';
import { GravitasConfig } from '../config/gravitasConfig';

export class ContextCollector {
    private resolver = new SymbolResolver();

    async init() {
        await this.resolver.init();
    }

    async search(query: string, config: GravitasConfig): Promise<string[]> {
        return new Promise((resolve, reject) => {
            const root = config.codebaseRoot;
            if (!root) return resolve([]);

            cp.exec(`rg -l "${query}" ${root}`, (err, stdout) => {
                if (err && err.code !== 1) return reject(err);
                if (!stdout) return resolve([]);
                resolve(stdout.split('\n').filter(f => f.trim() !== '').slice(0, 5));
            });
        });
    }

    async getContextForSymbol(file: string, symbol: string): Promise<string> {
        return this.resolver.resolve(file, symbol);
    }
}
