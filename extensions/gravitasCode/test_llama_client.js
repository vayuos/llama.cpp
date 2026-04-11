const { default: axios } = require('axios');
const config = { timeout: 30000, headers: { 'Content-Type': 'application/json' } };
const rawPath = '/home/viren/.gravitas/sockets/coder.sock';
config.socketPath = rawPath;
config.baseURL = 'http://localhost';

const client = axios.create(config);

(async () => {
    try {
        const response = await client.get('/health', { timeout: 2000 });
        console.log("SUCCESS:", response.data);
    } catch(e) {
        console.error("ERROR:", e.message);
    }
})();
