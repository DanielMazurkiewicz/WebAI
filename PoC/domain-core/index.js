const WebAI = require('../webai/index');
const operations = require('./operations');

const domain = 'core';
const dataType = 'fp32';

WebAI.defineCustomDomain(domain);

operations(domain, dataType);
