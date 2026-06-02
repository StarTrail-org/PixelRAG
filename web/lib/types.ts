export interface Query {
  text?: string;
  image?: string;
  embedding?: number[];
}

export interface SearchRequest {
  queries: Query[];
  n_docs?: number;
  nprobe?: number;
  min_tile_height?: number;
  instruction?: string;
  articles_only?: boolean;
}

export interface Hit {
  score: number;
  vector_id: number;
  article_id: number;
  tile_index: number;
  chunk_index: number;
  y_offset: number;
  tile_height: number;
  path: string;
  url: string;
}

export interface QueryResult {
  hits: Hit[];
}

export interface SearchResponse {
  results: QueryResult[];
}

export interface StatusResponse {
  total_vectors: number;
  dimension: number;
  nlist: number;
  nprobe: number;
  model: string;
  index_dir: string;
  tiles_dir: string;
  index_built_at: string;
  index_size_bytes: number;
  metadata_size_bytes: number;
}

export interface ArticleGroup {
  article_id: number;
  title: string;
  url: string;
  hits: (Hit & { rank: number })[];
}

export function groupHitsByArticle(hits: Hit[]): ArticleGroup[] {
  const map = new Map<number, ArticleGroup>();
  hits.forEach((hit, index) => {
    const ranked = { ...hit, rank: index + 1 };
    let group = map.get(hit.article_id);
    if (!group) {
      const raw = hit.url;
      const slug = raw.includes("/wiki/") ? raw.split("/wiki/").pop()! : raw;
      const title =
        decodeURIComponent(slug).replace(/_/g, " ") ||
        `Article #${hit.article_id}`;
      const url = raw.startsWith("http")
        ? raw
        : `https://en.wikipedia.org/wiki/${encodeURIComponent(slug)}`;
      group = {
        article_id: hit.article_id,
        title,
        url,
        hits: [],
      };
      map.set(hit.article_id, group);
    }
    group.hits.push(ranked);
  });
  return Array.from(map.values());
}
