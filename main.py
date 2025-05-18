#!/usr/bin/env python3
from audio_transcriptor import transcribe_audio
import split
import os
from joblib import Parallel, delayed
import pandas as pd
import json
from numpy import NaN
import subprocess
import ffmpeg
import shutil
import time
from joblib.externals.loky.process_executor import TerminatedWorkerError

def analyse_voice_msg(
        filepath: str, 
        model_size: str = "tiny", 
        language: str = "ru"
):
    # Настройки:
    chunk_length = 30
    min_length = 5
    
    duration = ffmpeg.probe(filepath)["format"]["duration"]
    if not float(duration) < min_length:
        if not float(duration) < chunk_length:
            output_dir = filepath[:-4] + "_chunks/"
            os.makedirs(output_dir, exist_ok=True)
            
            chunks = split.split_into_chunks(filepath, output_dir, chunk_length, min_length)
            print("Chunks:", chunks)
            
            max_retries = 3
            attempt = 0

            while attempt < max_retries:
                if not len(chunks) == 0:
                    try:
                        transcriptions = Parallel(n_jobs=len(chunks))(
                            delayed(transcribe_audio)(chunk, model_size, language)
                            for chunk in chunks
                        )
                        break
                    except TerminatedWorkerError as e:
                        attempt += 1
                        print(f"TerminatedWorkerError (attempt {attempt}/{max_retries}): {e}")
                        if attempt == max_retries:
                            print("Max retries reached. Skipping this file.")
                            transcriptions = [f"Ошибка {e}"]
                    except Exception as e:
                        print(f"Unexpected error during parallel transcription: {e}")
                        transcriptions = []
                        break
            else:
                transcriptions = []

            shutil.rmtree(output_dir)
        else:
            transcriptions = transcribe_audio(filepath, model_size, language)
        
        return "".join(transcriptions)
    else:
        return f"Голосовое сбщ <{min_length} секунд. Обработка пропущена."
        

def main():
    # Таймер
    start_time = time.time()
    
    # Настройки:
    directory = "input/ChatExport_2024/"
    transcription_model = "tiny"

    # Загрузка JSON
    with open(directory + "result.json", encoding="utf-8") as f:
        data = json.load(f)
        data = data["messages"]
    
    # Загрузка в Pandas DF
    df = pd.json_normalize(data)

    # df = df[df["media_type"] == 'voice_message']

    df["transcribed_voice_msg"] = df.apply(
        lambda row: analyse_voice_msg(filepath=directory + row["file"], model_size=transcription_model) if row["media_type"] == 'voice_message' else None,
        axis=1
    )

    df.to_csv(f"transcribed_{directory}.csv")
    print(df.head())
    print(f"Time taken to process {directory}: {time.time() - start_time}")

    



    



if __name__ == "__main__":
    main()
