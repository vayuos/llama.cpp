export interface GravitasConfig {
    codebaseRoot: string;

    coder: {
        binaryPath: string;
        modelPath: string;
        cudaDevices: string;
        endpoint: string;
        gpuLayers: number;
        threads: number;
        contextSize: number;
    };

    reviewer: {
        binaryPath: string;
        modelPath: string;
        endpoint: string;
        modelName: string;
        threads: number;
        strictMode: boolean;
    };
}
