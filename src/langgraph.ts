/**
 * LangGraph.js Checkpoint Saver & Cross-Thread Long-Term Store for HyperspaceDB.
 *
 * - `HyperspaceCheckpointSaver`: Persists LangGraph execution checkpoints with 801D Hybrid
 *   (33D Lorentz + 768D Cosine) state embeddings and automatic `STEP_TRANSITION` KAG lineage edges.
 * - `HyperspaceCrossThreadStore`: Namespace-hierarchical cross-thread memory store (`put`, `get`, `search`, `delete`).
 */

import * as crypto from "crypto";
import { HyperspaceClient } from "hyperspace-sdk-ts";

function deterministicUint32(key: string): number {
    const hash = crypto.createHash("sha256").update(key).digest();
    const val = hash.readUInt32BE(0) & 0x7fffffff;
    return val === 0 ? 1 : val;
}

function deterministicHybrid801d(text: string, stepDepth: number = 1): number[] {
    const digest = crypto.createHash("sha512").update(text).digest();
    const radial = Math.min(2.5, 0.25 * Math.max(1, stepDepth));
    const spatial32: number[] = [];
    for (let i = 0; i < 32; i++) {
        const b = digest[i % digest.length];
        spatial32.push(((b / 127.5) - 1.0) * (radial / Math.sqrt(32.0)));
    }
    const normSq = spatial32.reduce((acc, x) => acc + x * x, 0);
    const x0 = Math.sqrt(1.0 + normSq);
    const lorentz33 = [x0, ...spatial32];

    const tail768: number[] = [];
    for (let j = 0; j < 768; j++) {
        const b = digest[(j + 17) % digest.length];
        tail768.push((b / 127.5) - 1.0);
    }
    const tailNorm = Math.sqrt(tail768.reduce((acc, x) => acc + x * x, 0)) || 1.0;
    const normalizedTail = tail768.map((x) => x / tailNorm);
    return [...lorentz33, ...normalizedTail];
}

export interface CheckpointTuple {
    config: Record<string, any>;
    checkpoint: Record<string, any>;
    metadata: Record<string, any>;
    parentConfig?: Record<string, any>;
}

export class HyperspaceCheckpointSaver {
    private client: HyperspaceClient;
    private collectionName: string;
    private localCache = new Map<string, CheckpointTuple>();

    constructor(config: { client: HyperspaceClient; collectionName?: string }) {
        this.client = config.client;
        this.collectionName = config.collectionName ?? "langgraph_checkpoints";
    }

    async put(
        config: Record<string, any>,
        checkpoint: Record<string, any>,
        metadata: Record<string, any> = {}
    ): Promise<Record<string, any>> {
        const configurable = config?.configurable || {};
        const threadId = String(configurable.thread_id || "default");
        const checkpointNs = String(configurable.checkpoint_ns || "");
        const parentCheckpointId = String(configurable.checkpoint_id || "");
        const checkpointId = String(checkpoint.id || crypto.randomUUID());
        const step = Number(metadata?.step ?? 1);

        const compositeKey = `${threadId}:${checkpointNs}:${checkpointId}`;
        const pointId = deterministicUint32(compositeKey);
        const stateText = JSON.stringify({ checkpoint, metadata });
        const vec = deterministicHybrid801d(stateText, Math.max(1, step));

        const recordMeta: Record<string, string> = {
            thread_id: threadId,
            checkpoint_ns: checkpointNs,
            checkpoint_id: checkpointId,
            parent_checkpoint_id: parentCheckpointId,
            step: String(step),
            state_json: stateText,
        };

        await this.client.batchInsert(
            [{ id: pointId, vector: vec, metadata: recordMeta }],
            this.collectionName
        );

        if (parentCheckpointId) {
            const parentKey = `${threadId}:${checkpointNs}:${parentCheckpointId}`;
            const parentPointId = deterministicUint32(parentKey);
            try {
                await this.client.link(parentPointId, pointId, "STEP_TRANSITION", 1.0, this.collectionName);
            } catch {
                // Non-fatal
            }
        }

        const nextConfig = {
            configurable: {
                thread_id: threadId,
                checkpoint_ns: checkpointNs,
                checkpoint_id: checkpointId,
            },
        };

        this.localCache.set(compositeKey, {
            config: nextConfig,
            checkpoint,
            metadata,
            parentConfig: parentCheckpointId
                ? {
                      configurable: {
                          thread_id: threadId,
                          checkpoint_ns: checkpointNs,
                          checkpoint_id: parentCheckpointId,
                      },
                  }
                : undefined,
        });

        return nextConfig;
    }

    async getTuple(config: Record<string, any>): Promise<CheckpointTuple | undefined> {
        const configurable = config?.configurable || {};
        const threadId = String(configurable.thread_id || "default");
        const checkpointNs = String(configurable.checkpoint_ns || "");
        const checkpointId = configurable.checkpoint_id ? String(configurable.checkpoint_id) : undefined;

        if (checkpointId) {
            const key = `${threadId}:${checkpointNs}:${checkpointId}`;
            if (this.localCache.has(key)) {
                return this.localCache.get(key);
            }
        }

        const matches: Array<{ step: number; tuple: CheckpointTuple }> = [];
        for (const [key, val] of this.localCache.entries()) {
            if (key.startsWith(`${threadId}:${checkpointNs}:`)) {
                matches.push({ step: Number(val.metadata?.step ?? 0), tuple: val });
            }
        }
        if (matches.length > 0) {
            matches.sort((a, b) => b.step - a.step);
            return matches[0].tuple;
        }
        return undefined;
    }

    async traverseLineage(
        threadId: string,
        checkpointId: string,
        checkpointNs: string = "",
        maxDepth: number = 5
    ): Promise<any[]> {
        const pointId = deterministicUint32(`${threadId}:${checkpointNs}:${checkpointId}`);
        if (typeof (this.client as any).traverse === "function") {
            return (this.client as any).traverse(
                pointId,
                maxDepth,
                0.9,
                this.collectionName,
                "STEP_TRANSITION",
                false
            );
        }
        return [];
    }
}

export class HyperspaceCrossThreadStore {
    private client: HyperspaceClient;
    private collectionName: string;
    private cache = new Map<string, { namespace: string[]; key: string; value: Record<string, any> }>();

    constructor(config: { client: HyperspaceClient; collectionName?: string }) {
        this.client = config.client;
        this.collectionName = config.collectionName ?? "langgraph_long_term_store";
    }

    async put(namespace: string[], key: string, value: Record<string, any>): Promise<void> {
        const nsPath = namespace.join("/");
        const composite = `${nsPath}:${key}`;
        const pointId = deterministicUint32(composite);
        const vec = deterministicHybrid801d(JSON.stringify(value), Math.max(1, namespace.length));

        await this.client.batchInsert(
            [
                {
                    id: pointId,
                    vector: vec,
                    metadata: {
                        namespace: nsPath,
                        key,
                        value_json: JSON.stringify(value),
                    },
                },
            ],
            this.collectionName
        );
        this.cache.set(composite, { namespace, key, value });
    }

    async get(namespace: string[], key: string): Promise<Record<string, any> | undefined> {
        const composite = `${namespace.join("/")}:${key}`;
        return this.cache.get(composite)?.value;
    }

    async delete(namespace: string[], key: string): Promise<void> {
        const composite = `${namespace.join("/")}:${key}`;
        this.cache.delete(composite);
        const pointId = deterministicUint32(composite);
        await this.client.delete(pointId, this.collectionName);
    }

    async search(namespacePrefix: string[], query?: string, limit: number = 10): Promise<Array<{ namespace: string[]; key: string; value: Record<string, any>; score: number }>> {
        const prefix = namespacePrefix.join("/");
        if (query) {
            const qVec = deterministicHybrid801d(query, Math.max(1, namespacePrefix.length));
            const hits = await this.client.search(qVec, limit, this.collectionName, {
                filters: prefix ? [{ match: { key: "namespace", value: prefix } }] : undefined,
            });
            return hits.map((h: any) => {
                const meta = h.metadata || {};
                let parsed: Record<string, any> = {};
                try {
                    parsed = JSON.parse(meta.value_json || "{}");
                } catch {
                    parsed = meta;
                }
                return {
                    namespace: String(meta.namespace || prefix).split("/").filter(Boolean),
                    key: String(meta.key || h.id),
                    value: parsed,
                    score: Number(h.score ?? 1.0 / (1.0 + Number(h.distance ?? 0))),
                };
            });
        }

        const out: Array<{ namespace: string[]; key: string; value: Record<string, any>; score: number }> = [];
        for (const [k, item] of this.cache.entries()) {
            if (!prefix || k.startsWith(prefix)) {
                out.push({ ...item, score: 1.0 });
            }
        }
        return out.slice(0, limit);
    }
}
