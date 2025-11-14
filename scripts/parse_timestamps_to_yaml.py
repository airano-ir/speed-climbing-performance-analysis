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

    # Races that need adjustments
    races_need_end_extension = [1, 2, 7, 8, 12]  # +5s to end
    race_2_extra = 4  # Race 2 needs +4s (not +5s)
    races_late_start = [2, 13, 15, 23]  # Started late (need more buffer_before)

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

    # Post-processing: Fix end times and mark late starts
    for race in races:
        race_id = race['race_id']

        # Extend end times
        if race_id in races_need_end_extension:
            if race_id == 2:
                # Race 2 needs only +4s
                race['end_time'] = add_seconds_to_timestamp(race['end_time'], race_2_extra)
                race['notes'] = race.get('notes', '') + ' | End time extended +4s (early finish)' if race.get('notes') else 'End time extended +4s (early finish)'
            else:
                race['end_time'] = add_seconds_to_timestamp(race['end_time'], 5)
                race['notes'] = race.get('notes', '') + ' | End time extended +5s (early finish)' if race.get('notes') else 'End time extended +5s (early finish)'

        # Mark late starts (need more buffer)
        if race_id in races_late_start:
            race['late_start'] = True
            race['notes'] = race.get('notes', '') + ' | Late start (needs more buffer)' if race.get('notes') else 'Late start (needs more buffer)'

    config = {
        'video': {
            'name': 'Speed_finals_Villars_2024',
            'path': 'data/raw_videos/Speed_finals_Villars_2024.mp4',
            'location': 'Villars, Switzerland',
            'event_type': 'European Championship',
            'fps': 30.0,
        },
        'races': races,
        'notes': 'Auto belay malfunction in left lane caused rerun of all 1/8 final men races. Some races have late starts or early finishes.'
    }

    return config


def parse_chamonix_2024_data() -> Dict[str, Any]:
    """Parse Chamonix 2024 timestamps and athlete data."""

    # Races that need adjustments
    races_need_5s_extension = [1, 2, 4, 5, 6, 7, 11, 14, 15, 18, 19, 20, 21, 26, 29, 32]
    race_30_needs_8s = True  # Race 30 needs +8s
    races_late_start = [20, 26]  # Started late (need more buffer_before)

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

    # Post-processing: Fix end times and mark late starts
    for race in races:
        race_id = race['race_id']

        # Extend end times
        if race_id in races_need_5s_extension:
            race['end_time'] = add_seconds_to_timestamp(race['end_time'], 5)
            race['notes'] = race.get('notes', '') + ' | End time extended +5s (early finish)' if race.get('notes') else 'End time extended +5s (early finish)'
        elif race_id == 30:
            # Race 30 needs +8s
            race['end_time'] = add_seconds_to_timestamp(race['end_time'], 8)
            race['notes'] = race.get('notes', '') + ' | End time extended +8s (early finish)' if race.get('notes') else 'End time extended +8s (early finish)'

        # Mark late starts (need more buffer)
        if race_id in races_late_start:
            race['late_start'] = True
            race['notes'] = race.get('notes', '') + ' | Late start (needs more buffer)' if race.get('notes') else 'Late start (needs more buffer)'

    config = {
        'video': {
            'name': 'Speed_finals_Chamonix_2024',
            'path': 'data/raw_videos/Speed_finals_Chamonix_2024.mp4',
            'location': 'Chamonix, France',
            'event_type': 'IFSC World Cup',
            'fps': 30.0,
        },
        'races': races,
        'notes': 'Some races have late starts or early finishes. Adjusted accordingly.'
    }

    return config


def parse_innsbruck_2024_data() -> Dict[str, Any]:
    """Parse Innsbruck 2024 timestamps and athlete data."""

    # Note: Timestamps with "~" indicate late/approximate start times
    races_late_start = [2, 4, 6, 8, 9, 10, 12, 14, 15, 16, 17, 20, 21, 23, 24, 25, 27, 32]

    # Races that need end_time adjustments
    races_need_5s_extension = [3, 10, 11, 18, 23, 30]  # +5s to end
    race_2_needs_8s = True  # Race 2 needs +8s (special case)

    races = [
        # 1/8 final - Men (Races 1-8)
        {'race_id': 1, 'round': '1/8 final - Men', 'start_time': '14:55', 'end_time': '15:14',
         'athletes': {'left': {'name': 'Kevin Amon', 'country': 'Austria', 'bib_color': 'red/white'},
                      'right': {'name': 'Lorenz', 'country': 'Austria', 'bib_color': 'red/white'}}},
        {'race_id': 2, 'round': '1/8 final - Men', 'start_time': '15:58', 'end_time': '16:19',
         'athletes': {'left': {'name': 'Kostiantyn Pavlenko', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Joshua Bruyns', 'country': 'Germany', 'bib_color': 'black/red/yellow'}}},
        {'race_id': 3, 'round': '1/8 final - Men', 'start_time': '17:25', 'end_time': '17:30',
         'athletes': {'left': {'name': 'Yaroslav Tkach', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Paco Leman', 'country': 'France', 'bib_color': 'blue/white/red'}}},
        {'race_id': 4, 'round': '1/8 final - Men', 'start_time': '18:39', 'end_time': '19:01',
         'athletes': {'left': {'name': 'Leander', 'country': 'Austria', 'bib_color': 'red/white'},
                      'right': {'name': 'Alejandro Rabadan', 'country': 'Spain', 'bib_color': 'red/yellow'}}},
        {'race_id': 5, 'round': '1/8 final - Men', 'start_time': '19:49', 'end_time': '20:01',
         'athletes': {'left': {'name': 'Hryhoriy', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Marco', 'country': 'Italy', 'bib_color': 'blue/white/red'}}},
        {'race_id': 6, 'round': '1/8 final - Men', 'start_time': '20:56', 'end_time': '21:04',
         'athletes': {'left': {'name': 'Raife Hand', 'country': 'United Kingdom', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Andy Goodall', 'country': 'United Kingdom', 'bib_color': 'blue/white/red'}}},
        {'race_id': 7, 'round': '1/8 final - Men', 'start_time': '22:05', 'end_time': '22:15',
         'athletes': {'left': {'name': 'Miguel Barrios', 'country': 'Spain', 'bib_color': 'red/yellow'},
                      'right': {'name': 'Lucas Knapp', 'country': 'Austria', 'bib_color': 'red/white'}}},
        {'race_id': 8, 'round': '1/8 final - Men', 'start_time': '23:18', 'end_time': '23:25',
         'athletes': {'left': {'name': 'Luka', 'country': 'Italy', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Maxime', 'country': 'France', 'bib_color': 'blue/white/red'}}},

        # 1/8 final - Women (Races 9-16)
        {'race_id': 9, 'round': '1/8 final - Women', 'start_time': '26:36', 'end_time': '26:42',
         'athletes': {'left': {'name': 'Daria Tkacheva', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Emma Glode', 'country': 'Germany', 'bib_color': 'black/red/yellow'}}},
        {'race_id': 10, 'round': '1/8 final - Women', 'start_time': '27:44', 'end_time': '27:51',
         'athletes': {'left': {'name': 'Erica', 'country': 'Italy', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Nelly', 'country': 'United Kingdom', 'bib_color': 'blue/white/red'}}},
        {'race_id': 11, 'round': '1/8 final - Women', 'start_time': '28:49', 'end_time': '29:00',
         'athletes': {'left': {'name': 'Maria', 'country': 'Poland', 'bib_color': 'red/white'},
                      'right': {'name': 'L...', 'country': 'Spain', 'bib_color': 'red/yellow'}}},
        {'race_id': 12, 'round': '1/8 final - Women', 'start_time': '30:08', 'end_time': '30:20',
         'athletes': {'left': {'name': 'Yulia', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Sofia', 'country': 'Germany', 'bib_color': 'black/red/yellow'}}},
        {'race_id': 13, 'round': '1/8 final - Women', 'start_time': '32:10', 'end_time': '32:22',
         'athletes': {'left': {'name': 'Carla Martinez', 'country': 'Spain', 'bib_color': 'red/yellow'},
                      'right': {'name': 'Alex', 'country': 'Austria', 'bib_color': 'red/white'}},
         'notes': 'After technical break'},
        {'race_id': 14, 'round': '1/8 final - Women', 'start_time': '33:26', 'end_time': '33:35',
         'athletes': {'left': {'name': 'Sarah', 'country': 'Italy', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Oksana', 'country': 'Ukraine', 'bib_color': 'blue/yellow'}}},
        {'race_id': 15, 'round': '1/8 final - Women', 'start_time': '34:28', 'end_time': '34:42',
         'athletes': {'left': {'name': 'Leslie', 'country': 'France', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Manon Schmidt', 'country': 'Germany', 'bib_color': 'black/red/yellow'}}},
        {'race_id': 16, 'round': '1/8 final - Women', 'start_time': '35:37', 'end_time': '35:50',
         'athletes': {'left': {'name': 'Tetiana', 'country': 'Austria', 'bib_color': 'red/white'},
                      'right': {'name': 'Agnes', 'country': 'Italy', 'bib_color': 'blue/white/red'}}},

        # Quarter finals - Men (Races 17-20)
        {'race_id': 17, 'round': 'Quarter final - Men', 'start_time': '39:04', 'end_time': '39:14',
         'athletes': {'left': {'name': 'Lorenz', 'country': 'Austria', 'bib_color': 'red/white'},
                      'right': {'name': 'Kostiantyn Pavlenko', 'country': 'Ukraine', 'bib_color': 'blue/yellow'}}},
        {'race_id': 18, 'round': 'Quarter final - Men', 'start_time': '40:34', 'end_time': '40:43',
         'athletes': {'left': {'name': 'Paco Leman', 'country': 'France', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Alejandro Rabadan', 'country': 'Spain', 'bib_color': 'red/yellow'}}},
        {'race_id': 19, 'round': 'Quarter final - Men', 'start_time': '41:59', 'end_time': '42:08',
         'athletes': {'left': {'name': 'Andy Goodall', 'country': 'United Kingdom', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Hryhoriy', 'country': 'Ukraine', 'bib_color': 'blue/yellow'}}},
        {'race_id': 20, 'round': 'Quarter final - Men', 'start_time': '43:35', 'end_time': '43:47',
         'athletes': {'left': {'name': 'Lucas Knapp', 'country': 'Austria', 'bib_color': 'red/white'},
                      'right': {'name': 'Maxime', 'country': 'France', 'bib_color': 'blue/white/red'}}},

        # Quarter finals - Women (Races 21-24)
        {'race_id': 21, 'round': 'Quarter final - Women', 'start_time': '53:28', 'end_time': '53:35',
         'athletes': {'left': {'name': 'Daria', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Erica', 'country': 'Italy', 'bib_color': 'blue/white/red'}},
         'notes': 'After long technical break'},
        {'race_id': 22, 'round': 'Quarter final - Women', 'start_time': '54:48', 'end_time': '55:01',
         'athletes': {'left': {'name': 'Maria', 'country': 'Poland', 'bib_color': 'red/white'},
                      'right': {'name': 'Yulia', 'country': 'Germany', 'bib_color': 'black/red/yellow'}}},
        {'race_id': 23, 'round': 'Quarter final - Women', 'start_time': '56:24', 'end_time': '56:30',
         'athletes': {'left': {'name': 'Carla', 'country': 'Spain', 'bib_color': 'red/yellow'},
                      'right': {'name': 'Sarah', 'country': 'Italy', 'bib_color': 'blue/white/red'}}},
        {'race_id': 24, 'round': 'Quarter final - Women', 'start_time': '58:05', 'end_time': '58:15',
         'athletes': {'left': {'name': 'Leslie', 'country': 'France', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Agnes', 'country': 'Italy', 'bib_color': 'blue/white/red'}}},

        # Semi finals - Men (Races 25-26)
        {'race_id': 25, 'round': 'Semi final - Men', 'start_time': '01:04:57', 'end_time': '01:05:09',
         'athletes': {'left': {'name': 'Kostiantyn Pavlenko', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Alejandro Rabadan', 'country': 'Spain', 'bib_color': 'red/yellow'}}},
        {'race_id': 26, 'round': 'Semi final - Men', 'start_time': '01:07:40', 'end_time': '01:07:49',
         'athletes': {'left': {'name': 'Hryhoriy', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Lucas Knapp', 'country': 'Austria', 'bib_color': 'red/white'}}},

        # Semi finals - Women (Races 27-28)
        {'race_id': 27, 'round': 'Semi final - Women', 'start_time': '01:10:21', 'end_time': '01:10:29',
         'athletes': {'left': {'name': 'Daria', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Maria', 'country': 'Poland', 'bib_color': 'red/white'}}},
        {'race_id': 28, 'round': 'Semi final - Women', 'start_time': '01:12:22', 'end_time': '01:12:34',
         'athletes': {'left': {'name': 'Leslie', 'country': 'France', 'bib_color': 'blue/white/red'},
                      'right': {'name': 'Sarah', 'country': 'Italy', 'bib_color': 'blue/white/red'}}},

        # Small finals (Bronze) (Races 29-30)
        {'race_id': 29, 'round': 'Small final - Men (Bronze)', 'start_time': '01:16:20', 'end_time': '01:16:32',
         'athletes': {'left': {'name': 'Alejandro Rabadan', 'country': 'Spain', 'bib_color': 'red/yellow'},
                      'right': {'name': 'Hryhoriy', 'country': 'Ukraine', 'bib_color': 'blue/yellow'}}},
        {'race_id': 30, 'round': 'Small final - Women (Bronze)', 'start_time': '01:18:41', 'end_time': '01:18:51',
         'athletes': {'left': {'name': 'Maria', 'country': 'Poland', 'bib_color': 'red/white'},
                      'right': {'name': 'Sarah', 'country': 'Italy', 'bib_color': 'blue/white/red'}}},

        # Finals (Gold) (Races 31-32)
        {'race_id': 31, 'round': 'Final - Men (Gold)', 'start_time': '01:21:55', 'end_time': '01:22:01',
         'athletes': {'left': {'name': 'Kostiantyn Pavlenko', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Lucas Knapp', 'country': 'Austria', 'bib_color': 'red/white'}}},
        {'race_id': 32, 'round': 'Final - Women (Gold)', 'start_time': '01:24:34', 'end_time': '01:24:43',
         'athletes': {'left': {'name': 'Daria', 'country': 'Ukraine', 'bib_color': 'blue/yellow'},
                      'right': {'name': 'Leslie', 'country': 'France', 'bib_color': 'blue/white/red'}}},
    ]

    # Post-processing: Fix timestamps and mark late starts
    for race in races:
        race_id = race['race_id']

        # Race 2 special case: +20s to start, +8s to end
        if race_id == 2:
            race['start_time'] = add_seconds_to_timestamp(race['start_time'], 20)
            race['end_time'] = add_seconds_to_timestamp(race['end_time'], 8)
            race['notes'] = race.get('notes', '') + ' | Start time +20s, End time +8s (timing adjustments)' if race.get('notes') else 'Start time +20s, End time +8s (timing adjustments)'
        # Other races needing +5s to end
        elif race_id in races_need_5s_extension:
            race['end_time'] = add_seconds_to_timestamp(race['end_time'], 5)
            race['notes'] = race.get('notes', '') + ' | End time extended +5s (early finish)' if race.get('notes') else 'End time extended +5s (early finish)'

        # Mark late starts (need more buffer)
        if race_id in races_late_start:
            race['late_start'] = True
            race['notes'] = race.get('notes', '') + ' | Late start (needs more buffer)' if race.get('notes') else 'Late start (needs more buffer)'

    config = {
        'video': {
            'name': 'Speed_finals_Innsbruck_2024',
            'path': 'data/raw_videos/Speed_finals_Innsbruck_2024.mp4',
            'location': 'Innsbruck, Austria',
            'event_type': 'European Cup',
            'fps': 30.0,
        },
        'races': races,
        'notes': 'Many races have approximate/late start times. Some athlete names are incomplete (first or last name only). Commentary by Matthew Fall (British Speed Team member).'
    }

    return config


def parse_zilina_2025_data() -> Dict[str, Any]:
    """Parse Zilina 2025 timestamps and athlete data (European Youth Championships)."""

    races = [
        # 1/8 final - Women U17 (Races 1-8)
        {'race_id': 1, 'round': '1/8 final - Women U17', 'start_time': '14:58', 'end_time': '15:17',
         'athletes': {'left': {'name': 'Nina', 'country': 'Ukraine', 'bib_color': 'unknown'},
                      'right': {'name': 'Yevheniya', 'country': 'Ukraine', 'bib_color': 'unknown'}}},
        {'race_id': 2, 'round': '1/8 final - Women U17', 'start_time': '16:09', 'end_time': '16:25',
         'athletes': {'left': {'name': 'Rebecca', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Sela Novak', 'country': 'Slovenia', 'bib_color': 'unknown'}}},
        {'race_id': 3, 'round': '1/8 final - Women U17', 'start_time': '17:11', 'end_time': '17:27',
         'athletes': {'left': {'name': 'Lara', 'country': 'Austria', 'bib_color': 'unknown'},
                      'right': {'name': 'Anna', 'country': 'Ukraine', 'bib_color': 'unknown'}}},
        {'race_id': 4, 'round': '1/8 final - Women U17', 'start_time': '18:26', 'end_time': '18:47',
         'athletes': {'left': {'name': 'Gabrielle', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Nora Garcia', 'country': 'Spain', 'bib_color': 'unknown'}}},
        {'race_id': 5, 'round': '1/8 final - Women U17', 'start_time': '19:34', 'end_time': '19:53',
         'athletes': {'left': {'name': 'Emma', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Carolina', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 6, 'round': '1/8 final - Women U17 (Rerun)', 'start_time': '25:42', 'end_time': '26:03',
         'athletes': {'left': {'name': 'Zoya', 'country': 'Slovenia', 'bib_color': 'unknown'},
                      'right': {'name': 'Mia', 'country': 'Austria', 'bib_color': 'unknown'}},
         'notes': 'Rerun - both athletes fell in first attempt'},
        {'race_id': 7, 'round': '1/8 final - Women U17', 'start_time': '26:49', 'end_time': '27:09',
         'athletes': {'left': {'name': 'Alicia', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Unknown', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 8, 'round': '1/8 final - Women U17', 'start_time': '27:51', 'end_time': '28:09',
         'athletes': {'left': {'name': 'Unknown', 'country': 'Spain', 'bib_color': 'unknown'},
                      'right': {'name': 'Eda', 'country': 'Germany', 'bib_color': 'unknown'}}},

        # 1/8 final - Men U17 (Races 9-15)
        {'race_id': 9, 'round': '1/8 final - Men U17', 'start_time': '29:11', 'end_time': '29:23',
         'athletes': {'left': {'name': 'Leo', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Marc', 'country': 'Slovenia', 'bib_color': 'unknown'}}},
        {'race_id': 10, 'round': '1/8 final - Men U17', 'start_time': '30:13', 'end_time': '30:31',
         'athletes': {'left': {'name': 'Giorgio', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Davide', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 11, 'round': '1/8 final - Men U17', 'start_time': '31:50', 'end_time': '32:05',
         'athletes': {'left': {'name': 'Unknown', 'country': 'Germany', 'bib_color': 'unknown'},
                      'right': {'name': 'Dario', 'country': 'Spain', 'bib_color': 'unknown'}}},
        {'race_id': 12, 'round': '1/8 final - Men U17', 'start_time': '32:59', 'end_time': '33:12',
         'athletes': {'left': {'name': 'Ludovico', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Anthony', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 13, 'round': '1/8 final - Men U17', 'start_time': '34:15', 'end_time': '34:20',
         'athletes': {'left': {'name': 'Sean', 'country': 'Germany', 'bib_color': 'unknown'},
                      'right': {'name': 'Boris', 'country': 'Bulgaria', 'bib_color': 'unknown'}},
         'notes': 'False start by Sean - Boris wins by opponent error'},
        {'race_id': 14, 'round': '1/8 final - Men U17', 'start_time': '35:14', 'end_time': '35:36',
         'athletes': {'left': {'name': 'Felix', 'country': 'Germany', 'bib_color': 'unknown'},
                      'right': {'name': 'Jacobo', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 15, 'round': '1/8 final - Men U17', 'start_time': '37:10', 'end_time': '37:37',
         'athletes': {'left': {'name': 'Anton', 'country': 'Germany', 'bib_color': 'unknown'},
                      'right': {'name': 'Unknown', 'country': 'Germany', 'bib_color': 'unknown'}}},

        # 1/8 final - Women U19 (Races 16-23)
        {'race_id': 16, 'round': '1/8 final - Women U19', 'start_time': '38:31', 'end_time': '38:53',
         'athletes': {'left': {'name': 'Polina', 'country': 'Ukraine', 'bib_color': 'unknown'},
                      'right': {'name': 'Maria', 'country': 'Ukraine', 'bib_color': 'unknown'}}},
        {'race_id': 17, 'round': '1/8 final - Women U19', 'start_time': '39:41', 'end_time': '39:56',
         'athletes': {'left': {'name': 'Yohana', 'country': 'Austria', 'bib_color': 'unknown'},
                      'right': {'name': 'Martina', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 18, 'round': '1/8 final - Women U19', 'start_time': '40:41', 'end_time': '40:57',
         'athletes': {'left': {'name': 'Ksenia', 'country': 'Ukraine', 'bib_color': 'unknown'},
                      'right': {'name': 'Alyssa', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 19, 'round': '1/8 final - Women U19', 'start_time': '41:39', 'end_time': '42:04',
         'athletes': {'left': {'name': 'Eva', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Nina', 'country': 'Germany', 'bib_color': 'unknown'}}},
        {'race_id': 20, 'round': '1/8 final - Women U19', 'start_time': '42:36', 'end_time': '43:07',
         'athletes': {'left': {'name': 'Sara', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Carolina', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 21, 'round': '1/8 final - Women U19', 'start_time': '43:55', 'end_time': '44:06',
         'athletes': {'left': {'name': 'Natalia', 'country': 'Poland', 'bib_color': 'unknown'},
                      'right': {'name': 'Maria', 'country': 'Ukraine', 'bib_color': 'unknown'}}},
        {'race_id': 22, 'round': '1/8 final - Women U19', 'start_time': '44:46', 'end_time': '45:06',
         'athletes': {'left': {'name': 'Eva', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Nela', 'country': 'Germany', 'bib_color': 'unknown'}}},
        {'race_id': 23, 'round': '1/8 final - Women U19', 'start_time': '45:49', 'end_time': '46:08',
         'athletes': {'left': {'name': 'Unknown', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Eva', 'country': 'United Kingdom', 'bib_color': 'unknown'}}},

        # 1/8 final - Men U19 (Races 24-30)
        {'race_id': 24, 'round': '1/8 final - Men U19', 'start_time': '47:01', 'end_time': '47:16',
         'athletes': {'left': {'name': 'Paco Lehman', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Alex', 'country': 'Austria', 'bib_color': 'unknown'}}},
        {'race_id': 25, 'round': '1/8 final - Men U19', 'start_time': '47:58', 'end_time': '48:15',
         'athletes': {'left': {'name': 'Lucas', 'country': 'Austria', 'bib_color': 'unknown'},
                      'right': {'name': 'Titian', 'country': 'United Kingdom', 'bib_color': 'unknown'}}},
        {'race_id': 26, 'round': '1/8 final - Men U19', 'start_time': '50:02', 'end_time': '50:17',
         'athletes': {'left': {'name': 'Unknown', 'country': 'Unknown', 'bib_color': 'unknown'},
                      'right': {'name': 'Unknown', 'country': 'Unknown', 'bib_color': 'unknown'}},
         'notes': 'No athlete information in transcript'},
        {'race_id': 27, 'round': '1/8 final - Men U19', 'start_time': '50:58', 'end_time': '51:15',
         'athletes': {'left': {'name': 'Roberto', 'country': 'Spain', 'bib_color': 'unknown'},
                      'right': {'name': 'Alex', 'country': 'Ukraine', 'bib_color': 'unknown'}}},
        {'race_id': 28, 'round': '1/8 final - Men U19', 'start_time': '52:18', 'end_time': '52:35',
         'athletes': {'left': {'name': 'Unknown', 'country': 'Germany', 'bib_color': 'unknown'},
                      'right': {'name': 'Tomat', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 29, 'round': '1/8 final - Men U19', 'start_time': '53:29', 'end_time': '53:49',
         'athletes': {'left': {'name': 'Mighty', 'country': 'Switzerland', 'bib_color': 'unknown'},
                      'right': {'name': 'Colby', 'country': 'France', 'bib_color': 'unknown'}}},
        {'race_id': 30, 'round': '1/8 final - Men U19', 'start_time': '54:39', 'end_time': '55:00',
         'athletes': {'left': {'name': 'Unknown', 'country': 'Germany', 'bib_color': 'unknown'},
                      'right': {'name': 'Samuel', 'country': 'Italy', 'bib_color': 'unknown'}}},

        # Quarter finals - Women U17 (Races 31-34)
        {'race_id': 31, 'round': 'Quarter final - Women U17', 'start_time': '56:01', 'end_time': '56:17',
         'athletes': {'left': {'name': 'Nina', 'country': 'Ukraine', 'bib_color': 'unknown'},
                      'right': {'name': 'Rebecca', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 32, 'round': 'Quarter final - Women U17', 'start_time': '57:07', 'end_time': '57:19',
         'athletes': {'left': {'name': 'Lara', 'country': 'Austria', 'bib_color': 'unknown'},
                      'right': {'name': 'Gabrielle', 'country': 'France', 'bib_color': 'unknown'}}},
        {'race_id': 33, 'round': 'Quarter final - Women U17', 'start_time': '58:01', 'end_time': '58:23',
         'athletes': {'left': {'name': 'Carolina', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Mia', 'country': 'Austria', 'bib_color': 'unknown'}}},
        {'race_id': 34, 'round': 'Quarter final - Women U17', 'start_time': '59:30', 'end_time': '59:48',
         'athletes': {'left': {'name': 'Alicia', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Eda', 'country': 'Germany', 'bib_color': 'unknown'}}},

        # Quarter finals - Men U17 (Races 35-38)
        {'race_id': 35, 'round': 'Quarter final - Men U17', 'start_time': '01:00:42', 'end_time': '01:00:58',
         'athletes': {'left': {'name': 'Leo', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Unknown', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 36, 'round': 'Quarter final - Men U17', 'start_time': '01:01:52', 'end_time': '01:02:06',
         'athletes': {'left': {'name': 'Dario', 'country': 'Spain', 'bib_color': 'unknown'},
                      'right': {'name': 'Anthony', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 37, 'round': 'Quarter final - Men U17', 'start_time': '01:02:54', 'end_time': '01:03:09',
         'athletes': {'left': {'name': 'Boris', 'country': 'Bulgaria', 'bib_color': 'unknown'},
                      'right': {'name': 'Felix', 'country': 'Germany', 'bib_color': 'unknown'}}},
        {'race_id': 38, 'round': 'Quarter final - Men U17', 'start_time': '01:03:57', 'end_time': '01:04:02',
         'athletes': {'left': {'name': 'Sofia', 'country': 'Switzerland', 'bib_color': 'unknown'},
                      'right': {'name': 'Anton', 'country': 'Germany', 'bib_color': 'unknown'}},
         'notes': 'False start - Sofia wins by opponent error'},

        # Quarter finals - Women U19 (Races 39-42)
        {'race_id': 39, 'round': 'Quarter final - Women U19', 'start_time': '01:05:09', 'end_time': '01:05:28',
         'athletes': {'left': {'name': 'Polina', 'country': 'Ukraine', 'bib_color': 'unknown'},
                      'right': {'name': 'Martina', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 40, 'round': 'Quarter final - Women U19', 'start_time': '01:06:25', 'end_time': '01:06:35',
         'athletes': {'left': {'name': 'Ksenia', 'country': 'Ukraine', 'bib_color': 'unknown'},
                      'right': {'name': 'Evelina', 'country': 'France', 'bib_color': 'unknown'}}},
        {'race_id': 41, 'round': 'Quarter final - Women U19', 'start_time': '01:07:26', 'end_time': '01:07:41',
         'athletes': {'left': {'name': 'Sara', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Natalia', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 42, 'round': 'Quarter final - Women U19', 'start_time': '01:08:25', 'end_time': '01:08:41',
         'athletes': {'left': {'name': 'Eva', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Eva', 'country': 'United Kingdom', 'bib_color': 'unknown'}}},

        # Quarter finals - Men U19 (Races 43-46)
        {'race_id': 43, 'round': 'Quarter final - Men U19', 'start_time': '01:09:44', 'end_time': '01:10:02',
         'athletes': {'left': {'name': 'Paco Lehman', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Titian', 'country': 'United Kingdom', 'bib_color': 'unknown'}}},
        {'race_id': 44, 'round': 'Quarter final - Men U19', 'start_time': '01:10:57', 'end_time': '01:11:15',
         'athletes': {'left': {'name': 'Aidan', 'country': 'Germany', 'bib_color': 'unknown'},
                      'right': {'name': 'Unknown', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 45, 'round': 'Quarter final - Men U19', 'start_time': '01:12:04', 'end_time': '01:12:29',
         'athletes': {'left': {'name': 'Alberto', 'country': 'Spain', 'bib_color': 'unknown'},
                      'right': {'name': 'Unknown', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 46, 'round': 'Quarter final - Men U19', 'start_time': '01:13:23', 'end_time': '01:13:50',
         'athletes': {'left': {'name': 'Mighty', 'country': 'Switzerland', 'bib_color': 'unknown'},
                      'right': {'name': 'Anton', 'country': 'Germany', 'bib_color': 'unknown'}}},

        # Quarter finals - Women U21 (Races 47-50)
        {'race_id': 47, 'round': 'Quarter final - Women U21', 'start_time': '01:15:01', 'end_time': '01:15:11',
         'athletes': {'left': {'name': 'Manon', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Lucy', 'country': 'Czech Republic', 'bib_color': 'unknown'}}},
        {'race_id': 48, 'round': 'Quarter final - Women U21', 'start_time': '01:15:59', 'end_time': '01:16:19',
         'athletes': {'left': {'name': 'Daria', 'country': 'Ukraine', 'bib_color': 'unknown'},
                      'right': {'name': 'Magdalena', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 49, 'round': 'Quarter final - Women U21', 'start_time': '01:17:02', 'end_time': '01:17:22',
         'athletes': {'left': {'name': 'Louise', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Catalina', 'country': 'Ukraine', 'bib_color': 'unknown'}}},
        {'race_id': 50, 'round': 'Quarter final - Women U21', 'start_time': '01:18:13', 'end_time': '01:18:27',
         'athletes': {'left': {'name': 'Alla', 'country': 'United Kingdom', 'bib_color': 'unknown'},
                      'right': {'name': 'Unknown', 'country': 'Italy', 'bib_color': 'unknown'}}},

        # Quarter finals - Men U21 (Races 51-52)
        {'race_id': 51, 'round': 'Quarter final - Men U21', 'start_time': '01:20:12', 'end_time': '01:20:43',
         'athletes': {'left': {'name': 'Unknown', 'country': 'Poland', 'bib_color': 'unknown'},
                      'right': {'name': 'Oscar', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 52, 'round': 'Quarter final - Men U21', 'start_time': '01:21:27', 'end_time': '01:21:47',
         'athletes': {'left': {'name': 'Marco', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Max', 'country': 'France', 'bib_color': 'unknown'}}},

        # Semi finals - Women U17 (Races 53-54)
        {'race_id': 53, 'round': 'Semi final - Women U17', 'start_time': '01:29:58', 'end_time': '01:30:16',
         'athletes': {'left': {'name': 'Rebecca', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Lara', 'country': 'Austria', 'bib_color': 'unknown'}}},
        {'race_id': 54, 'round': 'Semi final - Women U17', 'start_time': '01:31:22', 'end_time': '01:31:38',
         'athletes': {'left': {'name': 'Mia', 'country': 'Austria', 'bib_color': 'unknown'},
                      'right': {'name': 'Alicia', 'country': 'Italy', 'bib_color': 'unknown'}}},

        # Semi finals - Men U17 (Races 55-56)
        {'race_id': 55, 'round': 'Semi final - Men U17', 'start_time': '01:32:36', 'end_time': '01:32:53',
         'athletes': {'left': {'name': 'Leo', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Dario', 'country': 'Spain', 'bib_color': 'unknown'}}},
        {'race_id': 56, 'round': 'Semi final - Men U17', 'start_time': '01:33:56', 'end_time': '01:34:07',
         'athletes': {'left': {'name': 'Boris', 'country': 'Bulgaria', 'bib_color': 'unknown'},
                      'right': {'name': 'Sofia', 'country': 'Switzerland', 'bib_color': 'unknown'}}},

        # Semi finals - Women U19 (Races 57-58)
        {'race_id': 57, 'round': 'Semi final - Women U19', 'start_time': '01:35:11', 'end_time': '01:35:24',
         'athletes': {'left': {'name': 'Martina', 'country': 'Poland', 'bib_color': 'unknown'},
                      'right': {'name': 'Ksenia', 'country': 'Ukraine', 'bib_color': 'unknown'}}},
        {'race_id': 58, 'round': 'Semi final - Women U19', 'start_time': '01:36:14', 'end_time': '01:36:27',
         'athletes': {'left': {'name': 'Sara', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Eva', 'country': 'Italy', 'bib_color': 'unknown'}}},

        # Semi finals - Men U19 (Races 59-60)
        {'race_id': 59, 'round': 'Semi final - Men U19', 'start_time': '01:37:36', 'end_time': '01:37:48',
         'athletes': {'left': {'name': 'Titian', 'country': 'United Kingdom', 'bib_color': 'unknown'},
                      'right': {'name': 'Aidan', 'country': 'Germany', 'bib_color': 'unknown'}}},
        {'race_id': 60, 'round': 'Semi final - Men U19', 'start_time': '01:38:40', 'end_time': '01:38:58',
         'athletes': {'left': {'name': 'Tommaso', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Anton', 'country': 'Germany', 'bib_color': 'unknown'}}},

        # Semi finals - Women U21 (Race 61)
        {'race_id': 61, 'round': 'Semi final - Women U21', 'start_time': '01:41:04', 'end_time': '01:41:18',
         'athletes': {'left': {'name': 'Louise', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Unknown', 'country': 'Italy', 'bib_color': 'unknown'}}},

        # Semi finals - Men U21 (Races 62-63)
        {'race_id': 62, 'round': 'Semi final - Men U21', 'start_time': '01:42:12', 'end_time': '01:42:23',
         'athletes': {'left': {'name': 'Jerome', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Oscar', 'country': 'Poland', 'bib_color': 'unknown'}}},
        {'race_id': 63, 'round': 'Semi final - Men U21', 'start_time': '01:43:13', 'end_time': '01:43:27',
         'athletes': {'left': {'name': 'Marco', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Dennis', 'country': 'Ukraine', 'bib_color': 'unknown'}}},

        # Small finals (Bronze) (Races 64-68)
        {'race_id': 64, 'round': 'Small final - Women U17 (Bronze)', 'start_time': '01:44:24', 'end_time': '01:44:42',
         'athletes': {'left': {'name': 'Mia', 'country': 'Austria', 'bib_color': 'unknown'},
                      'right': {'name': 'Lara', 'country': 'Austria', 'bib_color': 'unknown'}}},
        {'race_id': 65, 'round': 'Small final - Men U17 (Bronze)', 'start_time': '01:45:39', 'end_time': '01:45:57',
         'athletes': {'left': {'name': 'Dario', 'country': 'Spain', 'bib_color': 'unknown'},
                      'right': {'name': 'Boris', 'country': 'Bulgaria', 'bib_color': 'unknown'}}},
        {'race_id': 66, 'round': 'Small final - Women U19 (Bronze)', 'start_time': '01:46:44', 'end_time': '01:47:03',
         'athletes': {'left': {'name': 'Martina', 'country': 'Poland', 'bib_color': 'unknown'},
                      'right': {'name': 'Eva', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 67, 'round': 'Small final - Men U19 (Bronze)', 'start_time': '01:48:02', 'end_time': '01:48:23',
         'athletes': {'left': {'name': 'Titian', 'country': 'United Kingdom', 'bib_color': 'unknown'},
                      'right': {'name': 'Anton', 'country': 'Germany', 'bib_color': 'unknown'}}},
        {'race_id': 68, 'round': 'Small final - Men U21 (Bronze) - Rerun', 'start_time': '02:01:37', 'end_time': '02:01:49',
         'athletes': {'left': {'name': 'Oscar', 'country': 'Poland', 'bib_color': 'unknown'},
                      'right': {'name': 'Dennis', 'country': 'Ukraine', 'bib_color': 'unknown'}},
         'notes': 'Rerun - both athletes fell simultaneously in first attempt'},

        # Finals (Gold) (Races 69-72)
        {'race_id': 69, 'round': 'Final - Women U17 (Gold)', 'start_time': '02:02:43', 'end_time': '02:03:04',
         'athletes': {'left': {'name': 'Rebecca', 'country': 'Italy', 'bib_color': 'unknown'},
                      'right': {'name': 'Alicia', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 70, 'round': 'Final - Men U17 (Gold)', 'start_time': '02:03:30', 'end_time': '02:03:49',
         'athletes': {'left': {'name': 'Leo', 'country': 'France', 'bib_color': 'unknown'},
                      'right': {'name': 'Sofia', 'country': 'Switzerland', 'bib_color': 'unknown'}}},
        {'race_id': 71, 'round': 'Final - Women U19 (Gold)', 'start_time': '02:04:31', 'end_time': '02:04:47',
         'athletes': {'left': {'name': 'Ksenia', 'country': 'Ukraine', 'bib_color': 'unknown'},
                      'right': {'name': 'Sara', 'country': 'Italy', 'bib_color': 'unknown'}}},
        {'race_id': 72, 'round': 'Final - Men U19 (Gold)', 'start_time': '02:05:08', 'end_time': '02:05:28',
         'athletes': {'left': {'name': 'Aidan', 'country': 'Germany', 'bib_color': 'unknown'},
                      'right': {'name': 'Tommaso', 'country': 'Italy', 'bib_color': 'unknown'}}},
    ]

    config = {
        'video': {
            'name': 'Speed_finals_Zilina_2025',
            'path': 'data/raw_videos/Speed_finals_Zilina_2025.mp4',
            'location': 'Zilina, Slovakia',
            'event_type': 'European Youth Championships',
            'fps': 30.0,
        },
        'races': races,
        'notes': 'European Youth Championships with U17, U19, and U21 categories. Wall was very slippery causing many falls. Some athlete names incomplete. Finals for U21 Women and Men (races 73-74) have no timestamps in transcript. Notable champions: Leo (France U17) and Aidan (Germany U19) both World and European champions.'
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

    # Parse and save Innsbruck 2024
    print("\nGenerating Innsbruck 2024 config...")
    innsbruck_config = parse_innsbruck_2024_data()
    innsbruck_path = output_dir / 'innsbruck_2024.yaml'
    with open(innsbruck_path, 'w', encoding='utf-8') as f:
        yaml.dump(innsbruck_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] Saved: {innsbruck_path}")
    print(f"  Total races: {len(innsbruck_config['races'])}")

    # Parse and save Zilina 2025
    print("\nGenerating Zilina 2025 config...")
    zilina_config = parse_zilina_2025_data()
    zilina_path = output_dir / 'zilina_2025.yaml'
    with open(zilina_path, 'w', encoding='utf-8') as f:
        yaml.dump(zilina_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  [OK] Saved: {zilina_path}")
    print(f"  Total races: {len(zilina_config['races'])}")

    print("\n" + "="*60)
    print("Summary:")
    print(f"  Seoul 2024:     {len(seoul_config['races'])} races")
    print(f"  Villars 2024:   {len(villars_config['races'])} races")
    print(f"  Chamonix 2024:  {len(chamonix_config['races'])} races")
    print(f"  Innsbruck 2024: {len(innsbruck_config['races'])} races")
    print(f"  Zilina 2025:    {len(zilina_config['races'])} races")
    total = len(seoul_config['races']) + len(villars_config['races']) + len(chamonix_config['races']) + len(innsbruck_config['races']) + len(zilina_config['races'])
    print(f"  Total:          {total} races")
    print("="*60)


if __name__ == '__main__':
    main()
