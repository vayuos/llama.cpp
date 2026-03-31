import axios, { AxiosInstance } from 'axios';

export class LlamaHttpClient {
    private client: AxiosInstance;

    constructor(endpoint: string) {
        this.client = axios.create({
            baseURL: endpoint,
            timeout: 30000,
            headers: { 'Content-Type': 'application/json' }
        });
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
