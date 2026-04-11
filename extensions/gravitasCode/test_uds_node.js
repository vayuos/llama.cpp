const axios = require('axios');
axios.get('http://localhost/health', {
  socketPath: '/home/viren/.gravitas/sockets/coder.sock'
}).then(res => console.log(res.status, res.data)).catch(err => console.error(err.message));
