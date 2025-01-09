import logging
from slt.speech_retrieval1.speech_retrieval.pipeline import SpeechRetrievalPipeline, save_retrieved_chunks
import os

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration
    DATA_DIR = "path/to/data_dir"
    RESULTS_PATH = "./results"
    CONTEXT_PATH = "./home/snp2453/slt/speech_retrieval/context.mp3"
    QUERY_PATH = "./Question.mp3"
    OUTPUT_DIR = "./output"
    
    # Initialize pipeline
    pipeline = SpeechRetrievalPipeline(
        data_dir=DATA_DIR,
        results_path=RESULTS_PATH,
        chunk_duration=60
    )
    
    # Perform retrieval with both similarity metrics
    try:
        for metric in ['cosine', 'euclidean']:
            logger.info(f"\nPerforming retrieval using {metric} similarity...")
            retrieved_chunks = pipeline.retrieve(
                context_path=CONTEXT_PATH,
                query_path=QUERY_PATH,
                metric=metric,
                top_k=3
            )
            
            # Save results
            output_subdir = os.path.join(OUTPUT_DIR, f"{metric}_similarity")
            save_retrieved_chunks(retrieved_chunks, output_subdir)
            
            # Print results
            logger.info(f"\nTop chunks using {metric} similarity:")
            for chunk, similarity in retrieved_chunks:
                logger.info(
                    f"Chunk {chunk.start_time:.2f}s - {chunk.end_time:.2f}s: "
                    f"Similarity = {similarity:.4f}"
                )
                
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise

if __name__ == "__main__":
    main()