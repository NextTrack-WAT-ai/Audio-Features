from soundcloud_pipeline import SoundCloudPipeline, compare_results

pipeline = SoundCloudPipeline(start_index=1001, end_index=1040)

pipeline.download_songs()
# pipeline.analyze_songs()
