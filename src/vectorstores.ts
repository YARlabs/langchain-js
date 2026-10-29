import { VectorStore } from "@langchain/core/vectorstores";
import { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import * as crypto from "crypto";
import { HyperspaceClient } from "hyperspace-sdk-ts";

export interface HyperspaceStoreArgs {
    client: HyperspaceClient;
    collectionName?: string;
    enableDeduplication?: boolean;
    /**
     * If true, server will handle embedding (via insertText/searchText).
     * @default false
     */
    useServerSideEmbedding?: boolean;
    /**
     * @default 1536
     */
    dimension?: number;
    /**
     * @default "l2"
     */
    metric?: string;
    /**
     * Optional AES-256-GCM + Isometric client-side encryption key
     */
    encryptionKey?: string;
}

export class HyperspaceStore extends VectorStore {
    private client: HyperspaceClient;
    private collectionName: string;
    private enableDeduplication: boolean;
    private useServerSideEmbedding: boolean;
    private encryptionKey?: string;

    constructor(embeddings: Embeddings | undefined, args: HyperspaceStoreArgs) {
        super(embeddings || ({} as any), args);
        this.client = args.client;
        this.collectionName = args.collectionName ?? "default";
        this.enableDeduplication = args.enableDeduplication ?? true;
        this.useServerSideEmbedding = args.useServerSideEmbedding ?? false;
        this.encryptionKey = args.encryptionKey;

        if (this.encryptionKey) {
            this.client.registerCollectionKey(this.collectionName, this.encryptionKey, args.metric || "l2");
        }

        this._ensureCollection(args.dimension, args.metric);
    }

    private async _ensureCollection(desiredDim?: number, desiredMetric?: string) {
        try {
            const collections = (await this.client.listCollections()) as any[];
            const existing = collections.find((c: any) => c.name === this.collectionName);
            if (!existing && desiredDim && desiredMetric) {
                await this.client.createCollection(
                    this.collectionName,
                    {
                        components: [
                            {
                                name: "default",
                                metric: desiredMetric,
                                fullDimension: desiredDim,
                                weight: 1.0,
                            },
                        ],
                        cascadePipeline: [],
                        encrypted: !!this.encryptionKey,
                    } as any,
                    this.encryptionKey || ""
                );
            }
        } catch (e) {
            // Non-fatal during lazy initialization
        }
    }

    _vectorstoreType(): string {
        return "hyperspace";
    }

    async addDocuments(documents: Document[], options?: string[] | { ids?: string[] }): Promise<string[]> {
        const ids = Array.isArray(options) ? options : options?.ids;
        if (this.useServerSideEmbedding) {
            const resultIds: string[] = [];
            const implicitEdges: Array<{ sourceId: number; targetId: number; relation?: string; weight?: number }> = [];

            for (let i = 0; i < documents.length; i++) {
                const doc = documents[i];
                const idNum =
                    ids && ids[i]
                        ? parseInt(ids[i], 10) || this.computeContentHash(ids[i])
                        : this.computeContentHash(doc.pageContent);
                const metadata = { ...doc.metadata, text: doc.pageContent };

                const typedMetadata: Record<string, string> = {};
                for (const [key, value] of Object.entries(metadata)) {
                    if (key === "relations" && Array.isArray(value)) {
                        for (const rel of value) {
                            if (rel && rel.targetId !== undefined) {
                                implicitEdges.push({
                                    sourceId: idNum,
                                    targetId: Number(rel.targetId) || this.computeContentHash(String(rel.targetId)),
                                    relation: rel.relation || "RELATED_TO",
                                    weight: Number(rel.weight ?? 1.0),
                                });
                            }
                        }
                        continue;
                    }
                    typedMetadata[key] = typeof value === "object" ? JSON.stringify(value) : String(value);
                }

                await this.client.insertText(idNum, doc.pageContent, typedMetadata, this.collectionName);
                resultIds.push(idNum.toString());
            }

            if (implicitEdges.length > 0) {
                try {
                    await this.client.linkBatch(implicitEdges, this.collectionName);
                } catch {
                    // Non-fatal
                }
            }
            return resultIds;
        }

        if (!this.embeddings) {
            throw new Error("Embeddings required when useServerSideEmbedding is false");
        }

        const texts = documents.map(({ pageContent }) => pageContent);
        return this.addVectors(await this.embeddings.embedDocuments(texts), documents, { ids });
    }

    async addVectors(
        vectors: number[][],
        documents: Document[],
        options?: { ids?: string[]; metadatas?: Record<string, any>[] }
    ): Promise<string[]> {
        const ids = options?.ids || [];
        const resultIds: string[] = [];
        const metadatas = options?.metadatas || documents.map((d) => d.metadata);
        const items: Array<{ id: number; vector: number[]; metadata: Record<string, string> }> = [];
        const implicitEdges: Array<{ sourceId: number; targetId: number; relation?: string; weight?: number }> = [];

        for (let i = 0; i < vectors.length; i++) {
            const text = documents[i].pageContent;
            const vector = vectors[i];
            const metadata = metadatas[i] || {};

            let idStr = ids[i];
            let idNum: number;

            if (!idStr && this.enableDeduplication) {
                idNum = this.computeContentHash(text);
                idStr = idNum.toString();
            } else if (idStr) {
                idNum = parseInt(idStr, 10) || this.computeContentHash(idStr);
            } else {
                idNum = this.computeContentHash(`${text}:${i}`);
                idStr = idNum.toString();
            }

            const fullMetadata: Record<string, string> = {};
            for (const [key, value] of Object.entries(metadata)) {
                if (key === "relations" && Array.isArray(value)) {
                    for (const rel of value) {
                        const tgtRaw = rel?.targetId ?? rel?.target_id;
                        if (tgtRaw !== undefined) {
                            implicitEdges.push({
                                sourceId: idNum,
                                targetId: Number(tgtRaw) || this.computeContentHash(String(tgtRaw)),
                                relation: String(rel.relation || "RELATED_TO"),
                                weight: Number(rel.weight ?? 1.0),
                            });
                        }
                    }
                    continue;
                }
                if (value !== undefined && value !== null) {
                    fullMetadata[key] = typeof value === "object" ? JSON.stringify(value) : String(value);
                }
            }
            fullMetadata["text"] = text;
            fullMetadata["_doc_id"] = idStr;

            items.push({
                id: idNum,
                vector,
                metadata: fullMetadata,
            });
            resultIds.push(idStr);
        }

        await this.client.batchInsert(items, this.collectionName);

        if (implicitEdges.length > 0) {
            try {
                await this.client.linkBatch(implicitEdges, this.collectionName);
            } catch {
                // Non-fatal
            }
        }

        return resultIds;
    }

    async delete(params: { ids?: string[] }): Promise<void> {
        const ids = params.ids || [];
        if (ids.length === 0) return;
        const numIds = ids.map((idStr) => parseInt(idStr, 10) || this.computeContentHash(idStr));
        if (typeof (this.client as any).deleteBatch === "function") {
            await (this.client as any).deleteBatch(numIds, this.collectionName);
        } else {
            await Promise.all(numIds.map((idNum) => this.client.delete(idNum, this.collectionName)));
        }
    }

    async similaritySearchVectorWithScore(
        query: number[],
        k: number,
        filter?: Record<string, any> & { hybridAlpha?: number; hybridQuery?: string }
    ): Promise<[Document, number][]> {
        const { hybridAlpha, hybridQuery, ...cleanFilter } = filter || {};
        const filters = this.parseFilters(Object.keys(cleanFilter).length ? cleanFilter : undefined);
        const results = await this.client.search(query, k, this.collectionName, {
            filters,
            hybridAlpha,
            hybridQuery,
        });
        return this.resultsToDocuments(results);
    }

    async similaritySearch(
        query: string,
        k: number,
        filter?: Record<string, any> & { hybridAlpha?: number; hybridQuery?: string }
    ): Promise<Document[]> {
        if (this.useServerSideEmbedding) {
            const { hybridAlpha, ...cleanFilter } = filter || {};
            const filters = this.parseFilters(Object.keys(cleanFilter).length ? cleanFilter : undefined);
            const results = await this.client.searchText(query, k, this.collectionName, {
                filters,
                hybridAlpha,
            });
            const docsWithScore = this.resultsToDocuments(results);
            return docsWithScore.map(([doc]) => doc);
        }
        if (!this.embeddings) {
            throw new Error("Embeddings required when useServerSideEmbedding is false");
        }
        return super.similaritySearch(query, k, filter);
    }

    async similaritySearchWithScore(
        query: string,
        k: number,
        filter?: Record<string, any> & { hybridAlpha?: number; hybridQuery?: string }
    ): Promise<[Document, number][]> {
        if (this.useServerSideEmbedding) {
            const { hybridAlpha, ...cleanFilter } = filter || {};
            const filters = this.parseFilters(Object.keys(cleanFilter).length ? cleanFilter : undefined);
            const results = await this.client.searchText(query, k, this.collectionName, {
                filters,
                hybridAlpha,
            });
            return this.resultsToDocuments(results);
        }
        if (!this.embeddings || typeof (this.embeddings as any).embedQuery !== "function") {
            throw new Error("Embeddings required when useServerSideEmbedding is false");
        }
        return super.similaritySearchWithScore(query, k, filter);
    }

    async maxMarginalRelevanceSearch(
        query: string,
        options: { k: number; fetchK?: number; lambda?: number; filter?: Record<string, any> }
    ): Promise<Document[]> {
        if (this.useServerSideEmbedding || !this.embeddings) {
            return this.similaritySearch(query, options.k, options.filter);
        }

        const { k, fetchK = 20, filter } = options;
        const queryEmbedding = await this.embeddings.embedQuery(query);
        const filters = this.parseFilters(filter);

        const results = await this.client.search(queryEmbedding, fetchK, this.collectionName, { filters });
        return this.resultsToDocuments(results)
            .slice(0, k)
            .map(([doc]) => doc);
    }

    /**
     * Execute a 1-RPC Joint Vector-Graph Search combining dense/hyperbolic retrieval
     * with multi-hop CSR KAG graph traversal and optional Wave Resonance (`useWave: true`).
     */
    public async vectorGraphSearch(
        queryVector: number[],
        k: number = 5,
        options?: {
            maxDepth?: number;
            decayFactor?: number;
            relationFilter?: string;
            useWave?: boolean;
            filter?: Record<string, any>;
        }
    ): Promise<any[]> {
        const filters = this.parseFilters(options?.filter);
        return (this.client as any).vectorGraphSearch(
            queryVector,
            k,
            options?.maxDepth ?? 2,
            options?.decayFactor ?? 0.85,
            this.collectionName,
            {
                relationFilter: options?.relationFilter,
                filters,
                useWave: options?.useWave ?? false,
            }
        );
    }

    /**
     * Traverse explicit CSR KAG edges starting from `startNodeId`.
     */
    public async traverseGraph(
        startNodeId: number | string,
        options?: {
            maxDepth?: number;
            decayFactor?: number;
            relationFilter?: string;
            useWave?: boolean;
        }
    ): Promise<any[]> {
        const numId = typeof startNodeId === "number" ? startNodeId : parseInt(startNodeId, 10) || this.computeContentHash(startNodeId);
        return (this.client as any).traverse(
            numId,
            options?.maxDepth ?? 2,
            options?.decayFactor ?? 0.85,
            this.collectionName,
            options?.relationFilter,
            options?.useWave ?? false
        );
    }

    private parseFilters(filter?: Record<string, any>): any[] {
        if (!filter) return [];
        const filters: any[] = [];
        for (const [key, value] of Object.entries(filter)) {
            if (typeof value === "object" && value !== null) {
                if ("$in_ball" in value) {
                    filters.push({ ball: { key, ...(value.$in_ball as any) } });
                } else if ("$in_box" in value) {
                    filters.push({ box: { key, ...(value.$in_box as any) } });
                } else if ("$in_cone" in value) {
                    filters.push({ cone: { key, ...(value.$in_cone as any) } });
                } else {
                    const range: any = { key };
                    if ("$gte" in value) range.gte = value.$gte;
                    if ("$lte" in value) range.lte = value.$lte;
                    if ("$gt" in value) range.gt = value.$gt;
                    if ("$lt" in value) range.lt = value.$lt;
                    filters.push({ range });
                }
            } else {
                filters.push({ match: { key, value: String(value) } });
            }
        }
        return filters;
    }

    private resultsToDocuments(results: any[]): [Document, number][] {
        return results.map((r: any) => {
            const metadata = r.metadata || {};
            const text = metadata["text"] || "";
            const docMeta = { ...metadata };
            return [
                new Document({
                    pageContent: text,
                    metadata: docMeta,
                }),
                r.distance ?? r.score ?? 0,
            ];
        });
    }

    public computeContentHash(text: string): number {
        const hash = crypto.createHash("sha256").update(text).digest();
        const val = hash.readUInt32BE(0) & 0x7fffffff;
        return val === 0 ? 1 : val;
    }

    public async getCacheStats(): Promise<any> {
        return this.client.getCacheStats(this.collectionName);
    }

    public async clearCache(): Promise<boolean> {
        return this.client.clearCache(this.collectionName);
    }

    public async updateCacheConfig(policy: string, annThreshold?: number): Promise<boolean> {
        return this.client.updateCacheConfig(this.collectionName, policy, annThreshold);
    }

    /**
     * Creates an explicit directed edge between two entity nodes in the native Vector-Graph.
     */
    public async link(sourceId: number, targetId: number, relation: string = "OWNS", weight: number = 1.0): Promise<boolean> {
        return this.client.link(sourceId, targetId, relation, weight, this.collectionName);
    }

    /**
     * Batch-inserts explicit directed edges in CSR graph.
     */
    public async linkBatch(edges: Array<{ sourceId: number; targetId: number; relation?: string; weight?: number }>): Promise<number> {
        return this.client.linkBatch(edges, this.collectionName);
    }

    /**
     * Deletes an explicit directed edge with zero-lag invalidation.
     */
    public async unlink(sourceId: number, targetId: number, relation: string = "OWNS"): Promise<boolean> {
        return this.client.unlink(sourceId, targetId, relation, this.collectionName);
    }

    /**
     * Automatically induces explicit edges from existing metadata fields according to declarative rules.
     */
    public async induceEdgesFromMetadata(
        rules: Array<{ field: string; ruleType: string; targetField?: string; relation?: string; weight?: number }>
    ): Promise<number> {
        return this.client.induceEdgesFromMetadata(rules, this.collectionName);
    }

    /**
     * Tribunal Router: Mathematically verifies if LLM output stays within geometric context boundaries.
     */
    public async verifyGeneration(
        llmAnswer: string,
        contextIds: string[],
        maxDistanceThreshold?: number,
        checkGeodesic: boolean = true
    ): Promise<any> {
        return this.client.verifyGeneration(llmAnswer, contextIds, this.collectionName, maxDistanceThreshold, checkGeodesic);
    }

    static async fromTexts(
        texts: string[],
        metadatas: object[] | object,
        embeddings: Embeddings,
        dbConfig: HyperspaceStoreArgs
    ): Promise<HyperspaceStore> {
        const docs: Document[] = [];
        for (let i = 0; i < texts.length; i += 1) {
            const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
            const newDoc = new Document({
                pageContent: texts[i],
                metadata,
            });
            docs.push(newDoc);
        }
        const store = new HyperspaceStore(embeddings, dbConfig);
        await store.addDocuments(docs);
        return store;
    }
}
