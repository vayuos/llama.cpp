const axios = require('axios');
(async () => {
  try {
    const res = await axios({
      method: 'get',
      url: 'http://localhost/health',
      socketPath: '/home/viren/.gravitas/sockets/coder.sock'
    });
    console.log("SUCCESS:", res.status, res.data);
  } catch (err) {
    console.error("ERROR:", err.message);
  }
})();
