import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function applyPatch(patch: string) {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) throw new Error('No workspace folder open.');

    const rootPath = workspaceFolders[0].uri.fsPath;
    const tempPatchFile = path.join(rootPath, `.gravitas_patch_${Date.now()}.diff`);

    try {
        fs.writeFileSync(tempPatchFile, patch);
        const { stdout, stderr } = await execAsync(`patch -p1 -t < "${tempPatchFile}"`, { cwd: rootPath });
        console.log('Patch apply stdout:', stdout);
        if (stderr) console.warn('Patch apply stderr:', stderr);
        vscode.window.showInformationMessage('Gravitas: Patch applied successfully.');
    } catch (e: any) {
        throw new Error(`Failed to apply patch: ${e.message}`);
    } finally {
        if (fs.existsSync(tempPatchFile)) fs.unlinkSync(tempPatchFile);
    }
}
