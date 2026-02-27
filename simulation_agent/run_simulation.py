import hashlib
from yacs.config import CfgNode
import yaml
import torch
import numpy as np
import os
import sys
import time
from tqdm import tqdm

# Add parent directory to path to import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import paths

from simulator import Simulator, parse_args
from utils import utils
from utils.message import Message
from utils.event import update_event

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return CfgNode(config_dict)

def init_seed_agents(sim, seed_agents, target_movie):
    messages = []
    for agent_id in seed_agents:
        message = []
        agent = sim.agents[agent_id]
        name = agent.name
        contacts = sim.data.get_all_contacts(agent_id)
        if len(contacts) == 0:
            sim.logger.info(f"{name} (ID: {agent_id}) has no acquaintance.")
            message.append(
                Message(
                    agent_id=agent_id,
                    action="SOCIAL",
                    content=f"{name} has no acquaintance.",
                )
            )
            sim.round_msg.append(
                Message(
                    agent_id=agent_id,
                    action="SOCIAL",
                    content=f"{name} has no acquaintance.",
                )
            )
        
        sim.social_stat.post_num += 1
        sim.logger.info(f"{name} (ID: {agent_id}) is posting.")
        observation = f"I've just watched <{target_movie}> and it was amazing! I really want to recommend it to my friends."
        sim.logger.info(name + " posted: " + observation)
        if agent.event.action_type == "idle":
            agent.event = update_event(
                original_event=agent.event,
                start_time=sim.now,
                duration=0.1,
                target_agent=None,
                action_type="posting",
            )
        message.append(
            Message(
                agent_id=agent_id,
                action="POST",
                content=name + " posts: " + observation,
            )
        )
        sim.round_msg.append(
            Message(
                agent_id=agent_id,
                action="POST",
                content=name + " posts: " + observation,
            )
        )
        item_names = utils.extract_item_names(observation, "SOCIAL")
        for i in sim.agents.keys():
            if sim.agents[i].name in contacts:
                sim.agents[i].memory.add_memory(
                    name + " posts: " + observation, now=sim.now
                )
                sim.agents[i].update_heared_history(item_names)
                message.append(
                    Message(
                        agent_id=sim.agents[i].id,
                        action="POST",
                        content=sim.agents[i].name
                                + " observes that"
                                + name
                                + " posts: "
                                + observation,
                    )
                )
                sim.round_msg.append(
                    Message(
                        agent_id=sim.agents[i].id,
                        action="POST",
                        content=sim.agents[i].name
                                + " observes that"
                                + name
                                + " posts: "
                                + observation,
                    )
                )

        sim.logger.info(f"{contacts} get this post.")
        messages.append(message)
    return messages

def run_simulation(seed_size, target_movie_id, seed_agents):
    args = parse_args()
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    # create config
    config = CfgNode(new_allowed=True)
    output_file = os.path.join("output/data_generator", args.output_file)
    config = utils.add_variable_to_config(config, "output_file", output_file)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config = utils.add_variable_to_config(config, "recagent_memory", args.recagent_memory)
    config.merge_from_file(args.config_file)
    logger.info(f"\n{config}")
    os.environ["OPENAI_API_KEY"] = config["api_keys"][0]
    st = time.time()
    if config["simulator_restore_file_name"]:
        restore_path = os.path.join(config["simulator_dir"], config["simulator_restore_file_name"])
        sim = Simulator.restore(restore_path, config, logger)
        logger.info(f"Successfully Restore simulator from the file <{restore_path}>\n")
        logger.info(f"Start from the round {sim.round_cnt + 1}\n")
    else:
        sim = Simulator(config, logger)
        sim.load_simulator()
    if sim.config["social_random_k"] > 0:
        sim.clear_social()
        sim.add_social(sim.config["social_random_k"])
    print("Time for loading simulator: ", time.time() - st)
    sim.play()

    # Target Movie & Seed Agents
    target_movie = sim.data.items[target_movie_id]["name"]
    watch_count = seed_size
    watched_agents = set(seed_agents)
    newly_watched_agents = set(seed_agents)
    print(f"\n[Data Generator] Target Movie: <{target_movie}>")
    print(f"[Data Generator] Seeding {seed_size} agents: {seed_agents}")
    
    n_users = sim.data.get_user_num()
    
    # Get item embedding for target movie
    target_movie_emb = sim.recsys.model.item_content_emb[target_movie_id]
    target_movie_emb = target_movie_emb.cpu().detach().numpy()
    
    def calculate_array(user_content_emb, newly_watched_agents):
        sim_array = np.zeros(n_users)
        for user_id in range(n_users):
            user_emb = user_content_emb[user_id]
            similarity = np.dot(user_emb.flatten(), target_movie_emb.flatten()) / (np.linalg.norm(user_emb) * np.linalg.norm(target_movie_emb) + 1e-9)
            sim_array[user_id] = similarity
        X_array = np.zeros(n_users, dtype=bool)
        for user_id in newly_watched_agents:
            X_array[user_id] = True
        return X_array, sim_array
    
    sim_scores = np.zeros((config["round"]+1, n_users))
    X_flags = np.zeros((config["round"]+1, n_users), dtype=bool)
    X_flags[0], sim_scores[0] = calculate_array(sim.recsys.model.user_content_emb.cpu().detach().numpy(), newly_watched_agents)
    
    # Seed users post at the beginning of round 0
    messages = init_seed_agents(sim, seed_agents, target_movie)

    for i in tqdm(range(sim.round_cnt + 1, config["round"] + 1)):
        round_st = time.time()
        sim.round_cnt = sim.round_cnt + 1
        sim.logger.info(f"Round {sim.round_cnt}")
        sim.active_agents.clear()
        message = sim.round()
        messages.append(message)
        newly_watched_agents = set() # Track for this round
        for msgs in message:
            for m in msgs:
                if m.action == "RECOMMENDER" and "watches" in m.content:
                    names = utils.extract_item_names(m.content)
                    if len(names) == 0:
                        s = m.content
                        try:
                            start = s.find("watches ") + len("watches ")
                            end = s.rfind(".")
                            candidate = s[start:end].strip().strip("<>").strip('"').strip("'")
                            names = [candidate] if candidate else []
                        except Exception:
                            names = []
                    if target_movie in names:
                        newly_watched_agents.add(m.agent_id)
                        watched_agents.add(m.agent_id)
        X_flags[sim.round_cnt], sim_scores[sim.round_cnt] = calculate_array(sim.recsys.model.user_content_emb.cpu().detach().numpy(), newly_watched_agents)
        watch_count = len(watched_agents)
        newly_watch_count = len(newly_watched_agents)
        print(f"[Data Generator] Round {sim.round_cnt}: watchers of <{target_movie}> = {watch_count}")
        print(f"[Data Generator] Round {sim.round_cnt}: newly watchers of <{target_movie}> = {newly_watch_count}")

        print("Time for round: ", time.time() - round_st)
        print("Time for all: ", time.time() - st)
        sim.recsys.save_interaction()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[Data Generator] Simulation Finished.")
    
    # Explicitly release memory
    del sim
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X_flags, sim_scores

if __name__ == "__main__":
    # Load config to get relationship_path
    args = parse_args()

    target_movie_id = int(os.environ.get("TARGET_MOVIE_ID", "0"))
    with open("data/item.csv", "r") as file:
        target_movie = file.readlines()[target_movie_id+1].strip().split(",")[1]
    print(f"[Data Generator] Target Movie: <{target_movie}>")

    safe_movie_name = target_movie.replace(" ", "_").replace("/", "_")
    output_dir = paths.get_simulation_output_dir(target_movie)
    os.makedirs(output_dir, exist_ok=True)
    while True:
        seed_size = np.random.randint(1, 11)
        seed_agents = list(sorted(np.random.choice(range(1000), size=seed_size, replace=False)))
        seed_array = np.zeros(1000, dtype=bool)
        for agent_id in seed_agents:
            seed_array[agent_id] = True
        seed_str = str(seed_array)
        seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
        if f"{seed_hash}.npz" in os.listdir(output_dir):
            print(f"[Data Generator] Duplicate seed agents found, retrying...")
            continue
        break

    temp_base_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_base_dir, exist_ok=True)
    temp_dir = os.path.join(temp_base_dir, f"{seed_hash}")
    os.makedirs(temp_dir, exist_ok=True)
    while len(os.listdir(temp_dir)) < 10:
        X_k, sim_k = run_simulation(seed_size=seed_size, target_movie_id=target_movie_id, seed_agents=seed_agents)
        temp_file_path = os.path.join(temp_dir, f"{len(os.listdir(temp_dir))}.npz")
        np.savez(temp_file_path, sim=sim_k, X=X_k)
    sim_matrix = []
    X_matrix = []
    for file in os.listdir(temp_dir):
        npz_path = os.path.join(temp_dir, file)
        data = np.load(npz_path)
        X_matrix.append(data["X"])
        sim_matrix.append(data["sim"])
    X_matrix = np.array(X_matrix)
    sim_matrix = np.array(sim_matrix)
    output_file_path = os.path.join(output_dir, f"{seed_hash}.npz")
    np.savez(output_file_path, sim=sim_matrix, X=X_matrix)
    print(f"[Data Generator] Data saved to {output_file_path}")
    print(f"[Data Generator]   - sim shape: {sim_matrix.shape}")
    print(f"[Data Generator]   - X shape: {X_matrix.shape}")
