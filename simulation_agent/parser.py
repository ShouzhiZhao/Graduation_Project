import json
import re
from collections import defaultdict
import sys

import pandas as pd

# Dictionary to store agent data (agent ID -> {recommended_items: [list], item_details: [list]})
agent_data = defaultdict(lambda: {'recommended_items': [], 'item_details': []})
agent_mapping = [869, 657, 652, 883, 412, 16, 297, 135, 92, 396, 632, 525, 997, 451, 975, 843, 109, 375, 597, 531, 777,
                 47, 85, 379, 872, 542, 98, 206, 871, 227, 426, 332, 314, 819, 857, 170, 911, 892, 180, 934, 260, 395,
                 177, 879, 382, 870, 415, 856, 959, 211, 852, 929, 93, 779, 760, 765, 168, 246, 293, 232, 383, 752, 922,
                 913, 249, 334, 436, 546, 362, 210, 156, 229, 160, 719, 685, 544, 139, 536, 875, 537, 318, 524, 3, 795,
                 884, 145, 648, 912, 274, 534, 6, 960, 886, 106, 20, 565, 596, 429, 435, 7051]


def parse_log_line(line):
    global last_agent_id
    parts = line.split(' - ')
    if len(parts) < 5:
        return None  # Invalid log line
    action = parts[3]
    agent_part = parts[4]
    # Extract agent name and ID from the agent part
    try:
        agent_name_part, agent_id_part = agent_part.split(' [', 1)
        agent_id = agent_id_part.split('] ', 1)[0]
    except:
        return None

    if 'watches' in line:
        # Parse Agent Watched Item
        watched_item_match = re.search(r'watches .*?\[(.*?)\]$', line)
        if not watched_item_match:
            return None
        # agent_id = watched_item_match.group(1).strip()
        item_id = watched_item_match.group(1).strip()
        # item_name = watched_item.split(';;')[0].strip()
        # Retrieve recommended items from agent data
        # recommended_item_ids = agent_data.get(last_agent_id, {}).get('recommended_items', [])
        # recommended_item_titles = agent_data.get(last_agent_id, {}).get('item_titles', [])
        # for item_id, item in zip(recommended_item_ids, recommended_item_titles):
        #     if item == item_name:
        return {
            'Type': 'Watched Item',
            # 'Agent ID': agent_mapping[int(agent_id)],
            'Agent ID': agent_id,
            # 'Recommended Items': recommended_item_ids,
            # 'Watched Item': watched_item,
            'Item ID': item_id
        }

    elif action == 'create_agent':
        if 'rates the tags' in line:
            rating_json = line.split('rates the tags: ')[1]
            try:
                json.loads(rating_json)
                return {
                    'Type': 'Agent Creation',
                    'Agent ID': agent_id,
                    'Tag Rating': rating_json
                }
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in line: {line}")
        elif 'in group' in line:
            # Parse Agent Interest Update
            group_match = re.search(r' in group (\S+),', line)
            if not group_match:
                return None
            group = group_match.group(1)
            selected_interests_match = re.search(r'has selected new interests: (.*?)(?=\s*$)', line)
            if not selected_interests_match:
                return None
            selected_interests = selected_interests_match.group(1).strip()
            if selected_interests[0] == '[':
                selected_interests = ','.join(eval(selected_interests))
            return {
                'Type': 'Agent Creation',
                'Agent ID': agent_id,
                'Group': group,
                'Selected Interests': selected_interests
            }
    elif action == 'one_step':
        if 'is recommended' in line:
            # Parse Agent Recommendations
            item_ids_match = re.search(r'is recommended \[(.*?)\]', line)
            if not item_ids_match:
                return None
            item_ids = item_ids_match.group(1).strip()
            item_details_match = re.search(r'is recommended \[(.*?)\] \[\'(.*?)\'\]', line)
            if not item_details_match:
                return None
            score_match = re.search(r'score=(\d+)', line)
            if not score_match:
                score = None
            score = int(score_match.group(1).strip())
            item_details_str = item_details_match.group(2)
            # Parse item details
            items = item_details_str.split("', '")
            item_details = []
            for item in items:
                try:
                    title_part, description_part, _ = item.split(';;', 2)
                except:
                    return None
                title = title_part.strip()
                # genre = description_part.split(' Genre: ')[1].strip()
                item_details.append(title)
            # Store recommended items for this agent
            agent_data[agent_id]['recommended_items'] = item_ids.split(', ') if item_ids else []
            agent_data[agent_id]['item_titles'] = item_details
            return {
                'Type': 'Recommendation',
                'Agent ID': agent_id,
                'Recommended Items': item_ids.split(', ') if item_ids else [],
                'Item Titles': item_details,
                'Score': score
            }
    return None


def main(fn):
    for idx, line in enumerate(open(fn)):
        try:
            line = line.strip()
            parsed = parse_log_line(line)
            if parsed:
                yield parsed
        except Exception as e:
            raise ValueError('line {}: {}'.format(idx, e))


if __name__ == '__main__':
    df = pd.DataFrame(list(main('output/log/simulation_v3_upd_v2.log')))
    print(df)
    df.to_csv('output/log/simulation_v3_upd_v2.csv', index=False)
