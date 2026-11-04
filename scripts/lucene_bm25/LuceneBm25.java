import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/** Small standalone Lucene runner for the DPR 50k sample and sampled QA queries. */
public class LuceneBm25 {
    private static final float K1 = 0.9f;
    private static final float B = 0.4f;

    private static String decode(String value) {
        return new String(Base64.getDecoder().decode(value), StandardCharsets.UTF_8);
    }

    private static Query makeQuery(String text, Analyzer analyzer) throws IOException {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        int terms = 0;
        try (TokenStream stream = analyzer.tokenStream("contents", text)) {
            CharTermAttribute term = stream.addAttribute(CharTermAttribute.class);
            stream.reset();
            while (stream.incrementToken()) {
                builder.add(new org.apache.lucene.search.TermQuery(new Term("contents", term.toString())),
                        BooleanClause.Occur.SHOULD);
                terms++;
            }
            stream.end();
        }
        return terms == 0 ? new MatchNoDocsQuery() : builder.build();
    }

    private static void index(Path corpus, Path indexPath) throws IOException {
        Files.createDirectories(indexPath);
        try (Analyzer analyzer = new EnglishAnalyzer();
             Directory directory = FSDirectory.open(indexPath)) {
            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
            config.setSimilarity(new BM25Similarity(K1, B));
            try (IndexWriter writer = new IndexWriter(directory, config);
                 BufferedReader input = Files.newBufferedReader(corpus, StandardCharsets.UTF_8)) {
                String line;
                long count = 0;
                while ((line = input.readLine()) != null) {
                    String[] fields = line.split("\\t", 2);
                    if (fields.length != 2) throw new IOException("Malformed passage row " + (count + 1));
                    Document document = new Document();
                    document.add(new StringField("pid", fields[0], Field.Store.YES));
                    document.add(new TextField("contents", decode(fields[1]), Field.Store.YES));
                    writer.addDocument(document);
                    count++;
                }
                writer.commit();
                System.out.println("Indexed passages: " + count);
            }
        }
    }

    private static void search(Path queryFile, Path indexPath, Path output, int topK) throws IOException {
        try (Analyzer analyzer = new EnglishAnalyzer();
             Directory directory = FSDirectory.open(indexPath);
             DirectoryReader reader = DirectoryReader.open(directory);
             BufferedReader input = Files.newBufferedReader(queryFile, StandardCharsets.UTF_8);
             BufferedWriter result = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            searcher.setSimilarity(new BM25Similarity(K1, B));
            String line;
            long queryCount = 0;
            while ((line = input.readLine()) != null) {
                String[] fields = line.split("\\t", 2);
                if (fields.length != 2) throw new IOException("Malformed query row " + (queryCount + 1));
                Query query = makeQuery(decode(fields[1]), analyzer);
                TopDocs hits = searcher.search(query, topK);
                int rank = 0;
                for (ScoreDoc hit : hits.scoreDocs) {
                    Document document = searcher.doc(hit.doc);
                    result.write(fields[0] + "\t" + rank + "\t" + document.get("pid"));
                    result.newLine();
                    rank++;
                }
                queryCount++;
            }
            System.out.println("Searched queries: " + queryCount);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) throw new IllegalArgumentException("index <corpus.tsv> <index-dir> | search <queries.tsv> <index-dir> <hits.tsv> [topK]");
        if ("index".equals(args[0]) && args.length == 3) {
            index(Paths.get(args[1]), Paths.get(args[2]));
        } else if ("search".equals(args[0]) && (args.length == 4 || args.length == 5)) {
            int topK = args.length == 5 ? Integer.parseInt(args[4]) : 100;
            search(Paths.get(args[1]), Paths.get(args[2]), Paths.get(args[3]), topK);
        } else {
            throw new IllegalArgumentException("Invalid command arguments");
        }
    }
}
