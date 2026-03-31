import axios, { AxiosInstance } from 'axios';

export class LlamaHttpClient {
    private client: AxiosInstance;

    constructor(endpoint: string) {
        const isSocket = endpoint.startsWith('unix://');
        const config: any = {
            timeout: 300000, // 5 minute timeout for slow CPU inference
            headers: { 'Content-Type': 'application/json' }
        };

        if (isSocket) {
            config.socketPath = endpoint.replace('unix://', '');
            config.baseURL = 'http://localhost'; // Axios requires a dummy baseURL with socketPath
        } else {
            config.baseURL = endpoint;
        }

        this.client = axios.create(config);
    }

    async post(path: string, data: any): Promise<any> {
        const response = await this.client.post(path, data);
        return response.data;
    }

    async get(path: string): Promise<any> {
        const response = await this.client.get(path);
        return response.data;
    }
}
