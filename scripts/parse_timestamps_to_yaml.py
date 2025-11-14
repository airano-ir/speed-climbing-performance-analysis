#!/usr/bin/env python3
"""
Parse manual timestamps from text file to YAML config files.

This script reads the timestamp data from the text file and generates
YAML configuration files for each competition video.
"""

import re
import yaml
from pathlib import Path
from typing import List, Dict, Any


def parse_timestamp(timestamp_str: str) -> str:
    """
    Parse timestamp string to standardized format.

    Args:
        timestamp_str: Timestamp like "06:34" or "01:20:47"

    Returns:
        Standardized timestamp string
    """
    # Remove any markdown links
    timestamp_str = re.sub(r'\[\[([^\]]+)\]\]', r'\1', timestamp_str)
    timestamp_str = re.sub(r'\]\([^\)]+\)', '', timestamp_str)
    return timestamp_str.strip()


def add_seconds_to_timestamp(timestamp: str, seconds: int) -> str:
    """
    Add seconds to a timestamp.

    Args:
        timestamp: Time in "MM:SS" or "HH:MM:SS" format
        seconds: Seconds to add

    Returns:
        New timestamp string
    """
    parts = timestamp.strip().split(':')

    if len(parts) == 2:  # MM:SS
        minutes, secs = map(int, parts)
        total_secs = minutes * 60 + secs + seconds
        new_minutes = total_secs // 60
        new_secs = total_secs % 60
        return f"{new_minutes:02d}:{new_secs:02d}"
    elif len(parts) == 3:  # HH:MM:SS
        hours, minutes, secs = map(int, parts)
        total_secs = hours * 3600 + minutes * 60 + secs + seconds
        new_hours = total_secs // 3600
        new_minutes = (total_secs % 3600) // 60
        new_secs = total_secs % 60
        return f"{new_hours:02d}:{new_minutes:02d}:{new_secs:02d}"
    else:
        return timestamp


def parse_seoul_2024_data() -> Dict[str, Any]:
    """Parse Seoul 2024 timestamps and athlete data."""

    races = [
        # 1/8 نهایی زنان (مسابقات 1-8)
        {
            'race_id': 1, 'round': '1/8 final - Women',
            'start_time': '06:34', 'end_time': '06:38',
            'athletes': {
                'left': {'name': 'Raja Salilia', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Ren Masatu', 'country': 'Japan', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 2, 'round': '1/8 final - Women',
            'start_time': '08:12', 'end_time': '08:16',
            'athletes': {
                'left': {'name': 'Alveni Kadija', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Susan Nuri Adadya', 'country': 'Indonesia', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 3, 'round': '1/8 final - Women',
            'start_time': '09:53', 'end_time': '09:59',
            'athletes': {
                'left': {'name': 'Lijuan Deng', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Puja', 'country': 'Indonesia', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 4, 'round': '1/8 final - Women',
            'start_time': '11:31', 'end_time': '11:36',
            'athletes': {
                'left': {'name': 'Karen Hayeshi', 'country': 'Japan', 'bib_color': 'red/white'},
                'right': {'name': 'Shaokwin Zang', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 5, 'round': '1/8 final - Women',
            'start_time': '13:08', 'end_time': '13:13',
            'athletes': {
                'left': {'name': 'Zho Yayo', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Shim Hang', 'country': 'South Korea', 'bib_color': 'black/blue'}
            }
        },
        {
            'race_id': 6, 'round': '1/8 final - Women',
            'start_time': '14:36', 'end_time': '14:40',
            'athletes': {
                'left': {'name': 'Tamara', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'},
                'right': {'name': 'Nurl', 'country': 'Indonesia', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 7, 'round': '1/8 final - Women',
            'start_time': '16:04', 'end_time': '16:12',
            'athletes': {
                'left': {'name': 'Hanareum Song', 'country': 'South Korea', 'bib_color': 'black/blue'},
                'right': {'name': 'Jimin Jong', 'country': 'South Korea', 'bib_color': 'black/blue'}
            }
        },
        {
            'race_id': 8, 'round': '1/8 final - Women',
            'start_time': '17:33', 'end_time': '17:44',
            'athletes': {
                'left': {'name': 'Shenen Wang', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Sophia', 'country': 'USA', 'bib_color': 'blue/red/white'}
            }
        },

        # 1/8 نهایی مردان (مسابقات 9-16)
        {
            'race_id': 9, 'round': '1/8 final - Men',
            'start_time': '20:20', 'end_time': '20:26',
            'athletes': {
                'left': {'name': 'Kiromal Katibin', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Sebastian Lucke', 'country': 'Ecuador', 'bib_color': 'yellow/blue/red'}
            }
        },
        {
            'race_id': 10, 'round': '1/8 final - Men',
            'start_time': '21:38', 'end_time': '21:46',
            'athletes': {
                'left': {'name': 'Erik Noya', 'country': 'USA', 'bib_color': 'blue/red/white'},
                'right': {'name': 'Yaroslav Tkach', 'country': 'Ukraine', 'bib_color': 'blue/yellow'}
            }
        },
        {
            'race_id': 11, 'round': '1/8 final - Men',
            'start_time': '23:20', 'end_time': '23:27',
            'athletes': {
                'left': {'name': 'Peng Wu', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Aspar Jaelolo', 'country': 'Indonesia', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 12, 'round': '1/8 final - Men',
            'start_time': '24:45', 'end_time': '24:51',
            'athletes': {
                'left': {'name': 'Xinchang Wang', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Zamsa', 'country': 'Indonesia', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 13, 'round': '1/8 final - Men',
            'start_time': '26:12', 'end_time': '26:19',
            'athletes': {
                'left': {'name': 'Kostiantyn Pavlenko', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                'right': {'name': 'Veddriq Leonardo', 'country': 'Indonesia', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 14, 'round': '1/8 final - Men',
            'start_time': '27:50', 'end_time': '27:56',
            'athletes': {
                'left': {'name': 'Jun Yasukawa', 'country': 'Japan', 'bib_color': 'red/white'},
                'right': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'}
            }
        },
        {
            'race_id': 15, 'round': '1/8 final - Men',
            'start_time': '29:12', 'end_time': '29:12',
            'athletes': {
                'left': {'name': 'Michael Holm', 'country': 'USA', 'bib_color': 'blue/red/white'},
                'right': {'name': 'Sam Watson', 'country': 'USA', 'bib_color': 'blue/red/white'}
            },
            'notes': 'False start'
        },
        {
            'race_id': 16, 'round': '1/8 final - Men',
            'start_time': '30:33', 'end_time': '30:40',
            'athletes': {
                'left': {'name': 'Afriat', 'country': 'Thailand', 'bib_color': 'blue/red'},
                'right': {'name': 'Amir Maimuratov', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'}
            },
            'notes': 'Injury occurred'
        },

        # ربع نهایی زنان (مسابقات 17-20)
        {
            'race_id': 17, 'round': 'Quarter final - Women',
            'start_time': '34:22', 'end_time': '34:27',
            'athletes': {
                'left': {'name': 'Raja Salilia', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Susan Nuri Adadya', 'country': 'Indonesia', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 18, 'round': 'Quarter final - Women',
            'start_time': '35:57', 'end_time': '36:03',
            'athletes': {
                'left': {'name': 'Lijuan Deng', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Shaokwin Zang', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 19, 'round': 'Quarter final - Women',
            'start_time': '37:40', 'end_time': '37:46',
            'athletes': {
                'left': {'name': 'Tamara', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'},
                'right': {'name': 'Zho Yayo', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 20, 'round': 'Quarter final - Women',
            'start_time': '39:08', 'end_time': '39:13',
            'athletes': {
                'left': {'name': 'Jimin Jong', 'country': 'South Korea', 'bib_color': 'black/blue'},
                'right': {'name': 'Shenen Wang', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },

        # ربع نهایی مردان (مسابقات 21-24)
        {
            'race_id': 21, 'round': 'Quarter final - Men',
            'start_time': '54:29', 'end_time': '54:35',
            'athletes': {
                'left': {'name': 'Kiromal Katibin', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Yaroslav Tkach', 'country': 'Ukraine', 'bib_color': 'blue/yellow'}
            }
        },
        {
            'race_id': 22, 'round': 'Quarter final - Men',
            'start_time': '56:00', 'end_time': '56:05',
            'athletes': {
                'left': {'name': 'Peng Wu', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Xinchang Wang', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 23, 'round': 'Quarter final - Men',
            'start_time': '57:26', 'end_time': '57:30',
            'athletes': {
                'left': {'name': 'Veddriq Leonardo', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'}
            }
        },
        {
            'race_id': 24, 'round': 'Quarter final - Men',
            'start_time': '58:58', 'end_time': '59:04',
            'athletes': {
                'left': {'name': 'Amir Maimuratov', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'},
                'right': {'name': 'Michael Holm', 'country': 'USA', 'bib_color': 'blue/red/white'}
            }
        },

        # نیمه نهایی زنان (مسابقات 25-26)
        {
            'race_id': 25, 'round': 'Semi final - Women',
            'start_time': '01:07:08', 'end_time': '01:07:14',
            'athletes': {
                'left': {'name': 'Raja Salilia', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Lijuan Deng', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 26, 'round': 'Semi final - Women',
            'start_time': '01:09:01', 'end_time': '01:09:09',
            'athletes': {
                'left': {'name': 'Zho Yayo', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Shenen Wang', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },

        # نیمه نهایی مردان (مسابقات 27-28)
        {
            'race_id': 27, 'round': 'Semi final - Men',
            'start_time': '01:11:28', 'end_time': '01:11:35',
            'athletes': {
                'left': {'name': 'Kiromal Katibin', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Xinchang Wang', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 28, 'round': 'Semi final - Men',
            'start_time': '01:13:28', 'end_time': '01:13:32',
            'athletes': {
                'left': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Amir Maimuratov', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'}
            }
        },

        # فینال کوچک (مسابقات 29-30)
        {
            'race_id': 29, 'round': 'Small final - Women (Bronze)',
            'start_time': '01:16:57', 'end_time': '01:17:04',
            'athletes': {
                'left': {'name': 'Lijuan Deng', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Shenen Wang', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 30, 'round': 'Small final - Men (Bronze)',
            'start_time': '01:18:48', 'end_time': '01:18:54',
            'athletes': {
                'left': {'name': 'Kiromal Katibin', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'}
            }
        },

        # فینال (مسابقات 31-32)
        {
            'race_id': 31, 'round': 'Final - Women (Gold)',
            'start_time': '01:20:47', 'end_time': '01:20:53',
            'athletes': {
                'left': {'name': 'Raja Salilia', 'country': 'Indonesia', 'bib_color': 'red/white'},
                'right': {'name': 'Zho Yayo', 'country': 'China', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 32, 'round': 'Final - Men (Gold)',
            'start_time': '01:22:51', 'end_time': '01:22:57',
            'athletes': {
                'left': {'name': 'Xinchang Wang', 'country': 'China', 'bib_color': 'red/yellow'},
                'right': {'name': 'Amir Maimuratov', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'}
            }
        },
    ]

    # Post-processing: Fix end times and remove race 15
    # Races that need +5 seconds on end_time (finished early)
    races_need_extension = [1, 2, 3, 4, 5, 6, 7, 10, 13, 16, 17, 18, 20, 25, 26, 29, 30, 31, 32]

    for race in races:
        if race['race_id'] in races_need_extension:
            race['end_time'] = add_seconds_to_timestamp(race['end_time'], 5)
            if 'notes' not in race or not race['notes']:
                race['notes'] = 'End time extended +5s (race finished early)'
            else:
                race['notes'] += ' | End time extended +5s'

    # Remove race 15 (false start - too short)
    races = [r for r in races if r['race_id'] != 15]

    # Re-number races (keep original race_id for reference but sequential)
    for i, race in enumerate(races, 1):
        race['original_race_id'] = race['race_id']
        race['race_id'] = i

    config = {
        'video': {
            'name': 'Speed_finals_Seoul_2024',
            'path': 'data/raw_videos/Speed_finals_Seoul_2024.mp4',
            'location': 'Seoul, South Korea',
            'event_type': 'IFSC World Cup',
            'fps': 30.0,
        },
        'races': races,
        'notes': 'Race 15 (false start) removed. Some end times extended +5s due to early finish.'
    }

    return config


def parse_villars_2024_data() -> Dict[str, Any]:
    """Parse Villars 2024 timestamps and athlete data."""

    races = [
        # دور ۱/۸ نهایی مردان - Rerun (مسابقات 1-8)
        {
            'race_id': 1, 'round': '1/8 final - Men (Rerun)',
            'start_time': '51:10', 'end_time': '51:14',
            'athletes': {
                'left': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Masio Ganier', 'country': 'France', 'bib_color': 'blue/white/red'}
            }
        },
        {
            'race_id': 2, 'round': '1/8 final - Men (Rerun)',
            'start_time': '52:55', 'end_time': '53:00',
            'athletes': {
                'left': {'name': 'Kostiantyn Pavlenko', 'country': 'Ukraine', 'bib_color': 'black/blue/yellow'},
                'right': {'name': 'Kevin Ammon', 'country': 'France', 'bib_color': 'blue/white/red'}
            }
        },
        {
            'race_id': 3, 'round': '1/8 final - Men (Rerun)',
            'start_time': '54:52', 'end_time': '54:58',
            'athletes': {
                'left': {'name': 'Yaroslav Tkach', 'country': 'Ukraine', 'bib_color': 'black/blue/yellow'},
                'right': {'name': 'Miguel Gomez Barios', 'country': 'Spain', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 4, 'round': '1/8 final - Men (Rerun)',
            'start_time': '56:30', 'end_time': '56:35',
            'athletes': {
                'left': {'name': 'Erik Noya', 'country': 'Spain', 'bib_color': 'red/yellow'},
                'right': {'name': 'Gomo', 'country': 'Spain', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 5, 'round': '1/8 final - Men (Rerun)',
            'start_time': '58:16', 'end_time': '58:20',
            'athletes': {
                'left': {'name': 'Ludovico Fossali', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Alessandro Bulos', 'country': 'Italy', 'bib_color': 'blue'}
            }
        },
        {
            'race_id': 6, 'round': '1/8 final - Men (Rerun)',
            'start_time': '59:47', 'end_time': '59:56',
            'athletes': {
                'left': {'name': 'Marcin Dzienski', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Jan Luca Zoda', 'country': 'Italy', 'bib_color': 'blue'}
            },
            'notes': 'Fall occurred'
        },
        {
            'race_id': 7, 'round': '1/8 final - Men (Rerun)',
            'start_time': '01:01:28', 'end_time': '01:01:32',
            'athletes': {
                'left': {'name': 'Leander Carmans', 'country': 'Belgium', 'bib_color': 'black/red'},
                'right': {'name': 'Hii Otisian', 'country': 'Ukraine', 'bib_color': 'black/blue/yellow'}
            },
            'notes': 'Fall occurred'
        },
        {
            'race_id': 8, 'round': '1/8 final - Men (Rerun)',
            'start_time': '01:03:17', 'end_time': '01:03:23',
            'athletes': {
                'left': {'name': 'Sebastian Lucke', 'country': 'Germany', 'bib_color': 'black/red/yellow'},
                'right': {'name': 'Lucas Knap', 'country': 'Germany', 'bib_color': 'black/red/yellow'}
            }
        },

        # ربع نهایی زنان (مسابقات 9-12)
        {
            'race_id': 9, 'round': 'Quarter final - Women',
            'start_time': '01:05:56', 'end_time': '01:06:05',
            'athletes': {
                'left': {'name': 'Natalia Kalucka', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Liry', 'country': 'Spain', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 10, 'round': 'Quarter final - Women',
            'start_time': '01:07:24', 'end_time': '01:07:34',
            'athletes': {
                'left': {'name': 'Julia Randy', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Ka Videl Martinez', 'country': 'Spain', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 11, 'round': 'Quarter final - Women',
            'start_time': '01:08:51', 'end_time': '01:09:01',
            'athletes': {
                'left': {'name': 'Patricia Hujak', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Anna Brok', 'country': 'Poland', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 12, 'round': 'Quarter final - Women',
            'start_time': '01:10:38', 'end_time': '01:10:42',
            'athletes': {
                'left': {'name': 'Beatrice Colli', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Julia Cotch', 'country': 'Poland', 'bib_color': 'red/white'}
            }
        },

        # ربع نهایی مردان (مسابقات 13-16)
        {
            'race_id': 13, 'round': 'Quarter final - Men',
            'start_time': '01:12:29', 'end_time': '01:12:33',
            'athletes': {
                'left': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Kostiantyn Pavlenko', 'country': 'Ukraine', 'bib_color': 'black/blue/yellow'}
            }
        },
        {
            'race_id': 14, 'round': 'Quarter final - Men',
            'start_time': '01:14:16', 'end_time': '01:14:21',
            'athletes': {
                'left': {'name': 'Erik Noya', 'country': 'Spain', 'bib_color': 'red/yellow'},
                'right': {'name': 'Yaroslav Tkach', 'country': 'Ukraine', 'bib_color': 'black/blue/yellow'}
            },
            'notes': 'Slip occurred'
        },
        {
            'race_id': 15, 'round': 'Quarter final - Men',
            'start_time': '01:16:05', 'end_time': '01:16:10',
            'athletes': {
                'left': {'name': 'Ludovico Fossali', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Marcin Dzienski', 'country': 'Poland', 'bib_color': 'red/white'}
            },
            'notes': 'Slip occurred'
        },
        {
            'race_id': 16, 'round': 'Quarter final - Men',
            'start_time': '01:17:35', 'end_time': '01:17:42',
            'athletes': {
                'left': {'name': 'Leander Carmans', 'country': 'Belgium', 'bib_color': 'black/red'},
                'right': {'name': 'Sebastian Lucke', 'country': 'Germany', 'bib_color': 'black/red/yellow'}
            },
            'notes': 'Fall occurred'
        },

        # نیمه نهایی زنان (مسابقات 17-18)
        {
            'race_id': 17, 'round': 'Semi final - Women',
            'start_time': '01:24:54', 'end_time': '01:25:03',
            'athletes': {
                'left': {'name': 'Natalia Kalucka', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Julia Randy', 'country': 'Poland', 'bib_color': 'red/white'}
            },
            'notes': 'Slip occurred'
        },
        {
            'race_id': 18, 'round': 'Semi final - Women',
            'start_time': '01:26:30', 'end_time': '01:26:43',
            'athletes': {
                'left': {'name': 'Patricia Hujak', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Beatrice Colli', 'country': 'Italy', 'bib_color': 'blue'}
            },
            'notes': 'Fall and slip occurred'
        },

        # نیمه نهایی مردان (مسابقات 19-20)
        {
            'race_id': 19, 'round': 'Semi final - Men',
            'start_time': '01:29:23', 'end_time': '01:29:35',
            'athletes': {
                'left': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Erik Noya', 'country': 'Spain', 'bib_color': 'red/yellow'}
            },
            'notes': 'First sub-5 second time of competition'
        },
        {
            'race_id': 20, 'round': 'Semi final - Men',
            'start_time': '01:31:15', 'end_time': '01:31:25',
            'athletes': {
                'left': {'name': 'Ludovico Fossali', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Sebastian Lucke', 'country': 'Germany', 'bib_color': 'black/red/yellow'}
            },
            'notes': 'Fall and slip occurred'
        },

        # فینال کوچک (مسابقات 21-22)
        {
            'race_id': 21, 'round': 'Small final - Women (Bronze)',
            'start_time': '01:35:51', 'end_time': '01:36:03',
            'athletes': {
                'left': {'name': 'Julia Randy', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Beatrice Colli', 'country': 'Italy', 'bib_color': 'blue'}
            },
            'notes': 'Fall and slip occurred'
        },
        {
            'race_id': 22, 'round': 'Small final - Men (Bronze)',
            'start_time': '01:38:01', 'end_time': '01:38:07',
            'athletes': {
                'left': {'name': 'Erik Noya', 'country': 'Spain', 'bib_color': 'red/yellow'},
                'right': {'name': 'Sebastian Lucke', 'country': 'Germany', 'bib_color': 'black/red/yellow'}
            },
            'notes': 'False start'
        },

        # فینال (مسابقات 23-24)
        {
            'race_id': 23, 'round': 'Final - Women (Gold)',
            'start_time': '01:40:04', 'end_time': '01:40:12',
            'athletes': {
                'left': {'name': 'Natalia Kalucka', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Patricia Hujak', 'country': 'Poland', 'bib_color': 'red/white'}
            },
            'notes': 'Slip occurred'
        },
        {
            'race_id': 24, 'round': 'Final - Men (Gold)',
            'start_time': '01:42:10', 'end_time': '01:42:15',
            'athletes': {
                'left': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Ludovico Fossali', 'country': 'Italy', 'bib_color': 'blue'}
            },
            'notes': 'Slip occurred'
        },
    ]

    config = {
        'video': {
            'name': 'Speed_finals_Villars_2024',
            'path': 'data/raw_videos/Speed_finals_Villars_2024.mp4',
            'location': 'Villars, Switzerland',
            'event_type': 'European Championship',
            'fps': 30.0,
        },
        'races': races,
        'notes': 'Auto belay malfunction in left lane caused rerun of all 1/8 final men races'
    }

    return config


def parse_chamonix_2024_data() -> Dict[str, Any]:
    """Parse Chamonix 2024 timestamps and athlete data."""

    races = [
        # دور ۱/۸ نهایی زنان (مسابقات 1-8)
        {
            'race_id': 1, 'round': '1/8 final - Women',
            'start_time': '06:31', 'end_time': '06:37',
            'athletes': {
                'left': {'name': 'Natalia Kalucka', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Isis Rothfork', 'country': 'USA', 'bib_color': 'blue/white'}
            }
        },
        {
            'race_id': 2, 'round': '1/8 final - Women',
            'start_time': '08:02', 'end_time': '08:09',
            'athletes': {
                'left': {'name': 'Giulia Randi', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Franziska Ritter', 'country': 'Germany', 'bib_color': 'black/red/yellow'}
            }
        },
        {
            'race_id': 3, 'round': '1/8 final - Women',
            'start_time': '09:39', 'end_time': '09:48',
            'athletes': {
                'left': {'name': 'Jimin Jeong', 'country': 'South Korea', 'bib_color': 'black/blue'},
                'right': {'name': 'Karin Hayashi', 'country': 'Japan', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 4, 'round': '1/8 final - Women',
            'start_time': '11:21', 'end_time': '11:26',
            'athletes': {
                'left': {'name': 'Patrycja Chudziak', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Sophia Kirov', 'country': 'USA', 'bib_color': 'blue/white'}
            }
        },
        {
            'race_id': 5, 'round': '1/8 final - Women',
            'start_time': '12:52', 'end_time': '12:58',
            'athletes': {
                'left': {'name': 'Shengyan Wang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Agnese Rittosa', 'country': 'Italy', 'bib_color': 'blue'}
            }
        },
        {
            'race_id': 6, 'round': '1/8 final - Women',
            'start_time': '14:17', 'end_time': '14:25',
            'athletes': {
                'left': {'name': 'Manon Lebon', 'country': 'France', 'bib_color': 'blue/white'},
                'right': {'name': 'Capucine Viglione', 'country': 'France', 'bib_color': 'blue/white'}
            }
        },
        {
            'race_id': 7, 'round': '1/8 final - Women',
            'start_time': '16:00', 'end_time': '16:08',
            'athletes': {
                'left': {'name': 'Shaoqin Zhang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Ren Kayo Matsu', 'country': 'Japan', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 8, 'round': '1/8 final - Women',
            'start_time': '17:33', 'end_time': '17:38',
            'athletes': {
                'left': {'name': 'Leslie Romero', 'country': 'Spain', 'bib_color': 'red/yellow'},
                'right': {'name': 'Ya-Fei Niu', 'country': 'China', 'bib_color': 'red'}
            }
        },

        # دور ۱/۸ نهایی مردان (مسابقات 9-16)
        {
            'race_id': 9, 'round': '1/8 final - Men',
            'start_time': '20:23', 'end_time': '20:36',
            'athletes': {
                'left': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Damir Tursunov', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'}
            }
        },
        {
            'race_id': 10, 'round': '1/8 final - Men',
            'start_time': '22:02', 'end_time': '22:07',
            'athletes': {
                'left': {'name': 'Liang Zhang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Ryo Omasa', 'country': 'Japan', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 11, 'round': '1/8 final - Men',
            'start_time': '23:38', 'end_time': '23:46',
            'athletes': {
                'left': {'name': 'Xinchang Wang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Yong-Jun Jung', 'country': 'South Korea', 'bib_color': 'black/blue'}
            }
        },
        {
            'race_id': 12, 'round': '1/8 final - Men',
            'start_time': '25:20', 'end_time': '25:26',
            'athletes': {
                'left': {'name': 'Amir Maimuratov', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'},
                'right': {'name': 'Hryhorii Ilchyshyn', 'country': 'Ukraine', 'bib_color': 'blue/yellow'}
            }
        },
        {
            'race_id': 13, 'round': '1/8 final - Men',
            'start_time': '27:07', 'end_time': '27:11',
            'athletes': {
                'left': {'name': 'Jianguo Long', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Gian Luca Zodda', 'country': 'Italy', 'bib_color': 'blue'}
            }
        },
        {
            'race_id': 14, 'round': '1/8 final - Men',
            'start_time': '28:44', 'end_time': '28:51',
            'athletes': {
                'left': {'name': 'Yaroslav Tkach', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                'right': {'name': 'Erik Noya', 'country': 'Spain', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 15, 'round': '1/8 final - Men',
            'start_time': '30:19', 'end_time': '30:25',
            'athletes': {
                'left': {'name': 'Samuel Watson', 'country': 'USA', 'bib_color': 'blue/white'},
                'right': {'name': 'Shuto Hzz', 'country': 'Japan', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 16, 'round': '1/8 final - Men',
            'start_time': '32:07', 'end_time': '32:11',
            'athletes': {
                'left': {'name': 'Ludovico Fossali', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Pierre Rebreyend', 'country': 'France', 'bib_color': 'blue/white'}
            }
        },

        # ربع نهایی زنان (مسابقات 17-20)
        {
            'race_id': 17, 'round': 'Quarter final - Women',
            'start_time': '35:05', 'end_time': '35:10',
            'athletes': {
                'left': {'name': 'Natalia Kalucka', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Giulia Randi', 'country': 'Italy', 'bib_color': 'blue'}
            }
        },
        {
            'race_id': 18, 'round': 'Quarter final - Women',
            'start_time': '36:34', 'end_time': '36:41',
            'athletes': {
                'left': {'name': 'Jimin Jeong', 'country': 'South Korea', 'bib_color': 'black/blue'},
                'right': {'name': 'Patrycja Chudziak', 'country': 'Poland', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 19, 'round': 'Quarter final - Women',
            'start_time': '38:05', 'end_time': '38:11',
            'athletes': {
                'left': {'name': 'Shengyan Wang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Capucine Viglione', 'country': 'France', 'bib_color': 'blue/white'}
            }
        },
        {
            'race_id': 20, 'round': 'Quarter final - Women',
            'start_time': '39:43', 'end_time': '39:47',
            'athletes': {
                'left': {'name': 'Shaoqin Zhang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Leslie Romero', 'country': 'Spain', 'bib_color': 'red/yellow'}
            }
        },

        # ربع نهایی مردان (مسابقات 21-24)
        {
            'race_id': 21, 'round': 'Quarter final - Men',
            'start_time': '42:30', 'end_time': '42:37',
            'athletes': {
                'left': {'name': 'Matteo Zurloni', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Ryo Omasa', 'country': 'Japan', 'bib_color': 'red/white'}
            }
        },
        {
            'race_id': 22, 'round': 'Quarter final - Men',
            'start_time': '44:15', 'end_time': '44:19',
            'athletes': {
                'left': {'name': 'Xinchang Wang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Amir Maimuratov', 'country': 'Kazakhstan', 'bib_color': 'blue/yellow'}
            }
        },
        {
            'race_id': 23, 'round': 'Quarter final - Men',
            'start_time': '45:59', 'end_time': '46:05',
            'athletes': {
                'left': {'name': 'Gian Luca Zodda', 'country': 'Italy', 'bib_color': 'blue'},
                'right': {'name': 'Erik Noya', 'country': 'Spain', 'bib_color': 'red/yellow'}
            }
        },
        {
            'race_id': 24, 'round': 'Quarter final - Men',
            'start_time': '47:42', 'end_time': '47:47',
            'athletes': {
                'left': {'name': 'Samuel Watson', 'country': 'USA', 'bib_color': 'blue/white'},
                'right': {'name': 'Pierre Rebreyend', 'country': 'France', 'bib_color': 'blue/white'}
            }
        },

        # نیمه نهایی زنان (مسابقات 25-26)
        {
            'race_id': 25, 'round': 'Semi final - Women',
            'start_time': '54:15', 'end_time': '54:25',
            'athletes': {
                'left': {'name': 'Natalia Kalucka', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Jimin Jeong', 'country': 'South Korea', 'bib_color': 'black/blue'}
            },
            'notes': 'Slip and recovery'
        },
        {
            'race_id': 26, 'round': 'Semi final - Women',
            'start_time': '56:02', 'end_time': '56:06',
            'athletes': {
                'left': {'name': 'Shengyan Wang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Shaoqin Zhang', 'country': 'China', 'bib_color': 'red'}
            }
        },

        # نیمه نهایی مردان (مسابقات 27-28)
        {
            'race_id': 27, 'round': 'Semi final - Men',
            'start_time': '59:09', 'end_time': '59:15',
            'athletes': {
                'left': {'name': 'Ryo Omasa', 'country': 'Japan', 'bib_color': 'red/white'},
                'right': {'name': 'Xinchang Wang', 'country': 'China', 'bib_color': 'red'}
            }
        },
        {
            'race_id': 28, 'round': 'Semi final - Men',
            'start_time': '01:00:58', 'end_time': '01:01:03',
            'athletes': {
                'left': {'name': 'Erik Noya', 'country': 'Spain', 'bib_color': 'red/yellow'},
                'right': {'name': 'Samuel Watson', 'country': 'USA', 'bib_color': 'blue/white'}
            }
        },

        # فینال کوچک (مسابقات 29-30)
        {
            'race_id': 29, 'round': 'Small final - Women (Bronze)',
            'start_time': '01:06:04', 'end_time': '01:06:12',
            'athletes': {
                'left': {'name': 'Jimin Jeong', 'country': 'South Korea', 'bib_color': 'black/blue'},
                'right': {'name': 'Shengyan Wang', 'country': 'China', 'bib_color': 'red'}
            }
        },
        {
            'race_id': 30, 'round': 'Small final - Men (Bronze)',
            'start_time': '01:07:44', 'end_time': '01:07:51',
            'athletes': {
                'left': {'name': 'Ryo Omasa', 'country': 'Japan', 'bib_color': 'red/white'},
                'right': {'name': 'Erik Noya', 'country': 'Spain', 'bib_color': 'red/yellow'}
            }
        },

        # فینال (مسابقات 31-32)
        {
            'race_id': 31, 'round': 'Final - Women (Gold)',
            'start_time': '01:09:41', 'end_time': '01:09:48',
            'athletes': {
                'left': {'name': 'Natalia Kalucka', 'country': 'Poland', 'bib_color': 'red/white'},
                'right': {'name': 'Shaoqin Zhang', 'country': 'China', 'bib_color': 'red'}
            }
        },
        {
            'race_id': 32, 'round': 'Final - Men (Gold)',
            'start_time': '01:11:59', 'end_time': '01:12:04',
            'athletes': {
                'left': {'name': 'Xinchang Wang', 'country': 'China', 'bib_color': 'red'},
                'right': {'name': 'Samuel Watson', 'country': 'USA', 'bib_color': 'blue/white'}
            }
        },
    ]

    config = {
        'video': {
            'name': 'Speed_finals_Chamonix_2024',
            'path': 'data/raw_videos/Speed_finals_Chamonix_2024.mp4',
            'location': 'Chamonix, France',
            'event_type': 'IFSC World Cup',
            'fps': 30.0,
        },
        'races': races
    }

    return config


def main():
    """Generate YAML config files for all competitions."""

    # Create output directory
    output_dir = Path('configs/race_timestamps')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse and save Seoul 2024
    print("Generating Seoul 2024 config...")
    seoul_config = parse_seoul_2024_data()
    seoul_path = output_dir / 'seoul_2024.yaml'
    with open(seoul_path, 'w', encoding='utf-8') as f:
        yaml.dump(seoul_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] Saved: {seoul_path}")
    print(f"  Total races: {len(seoul_config['races'])}")

    # Parse and save Villars 2024
    print("\nGenerating Villars 2024 config...")
    villars_config = parse_villars_2024_data()
    villars_path = output_dir / 'villars_2024.yaml'
    with open(villars_path, 'w', encoding='utf-8') as f:
        yaml.dump(villars_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] Saved: {villars_path}")
    print(f"  Total races: {len(villars_config['races'])}")

    # Parse and save Chamonix 2024
    print("\nGenerating Chamonix 2024 config...")
    chamonix_config = parse_chamonix_2024_data()
    chamonix_path = output_dir / 'chamonix_2024.yaml'
    with open(chamonix_path, 'w', encoding='utf-8') as f:
        yaml.dump(chamonix_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] Saved: {chamonix_path}")
    print(f"  Total races: {len(chamonix_config['races'])}")

    print("\n" + "="*60)
    print("Summary:")
    print(f"  Seoul 2024:    {len(seoul_config['races'])} races")
    print(f"  Villars 2024:  {len(villars_config['races'])} races")
    print(f"  Chamonix 2024: {len(chamonix_config['races'])} races")
    print(f"  Total:         {len(seoul_config['races']) + len(villars_config['races']) + len(chamonix_config['races'])} races")
    print("="*60)


if __name__ == '__main__':
    main()
