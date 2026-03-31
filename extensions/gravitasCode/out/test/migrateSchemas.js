"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const registryPath = path.join(__dirname, '..', 'schemas', 'events.json');
const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
const baseProperties = {
    "eventId": { "type": "string" },
    "taskId": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" },
    "type": { "type": "string" },
    "attemptNo": { "type": "integer" },
    "phaseId": { "type": "string" }
};
const baseRequired = ["eventId", "taskId", "timestamp", "type"];
for (const [type, schema] of Object.entries(registry)) {
    const s = schema;
    if (!s.properties)
        s.properties = {};
    if (!s.required)
        s.required = [];
    // Merge properties
    for (const [prop, propSchema] of Object.entries(baseProperties)) {
        if (!s.properties[prop]) {
            s.properties[prop] = propSchema;
        }
    }
    // Merge required
    for (const req of baseRequired) {
        if (!s.required.includes(req)) {
            s.required.push(req);
        }
    }
    // Set const for type - Force overwrite to ensure alignment with key
    if (s.properties.type) {
        s.properties.type.const = type;
    }
    s.additionalProperties = false;
}
fs.writeFileSync(registryPath, JSON.stringify(registry, null, 4), 'utf8');
console.log('✅ Standardized 34 schemas in events.json');
//# sourceMappingURL=migrateSchemas.js.map