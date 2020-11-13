#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import argparse
import itertools

import textworld
import textworld.agents

from textworld.render.graph import build_graph_from_facts, show_graph
from textworld.logic import Proposition

import logging
import json
import sys

def build_parser():
    description = "Play a TextWorld game (.z8 or .ulx)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("game")
    parser.add_argument("--mode", default="human", metavar="MODE",
                        choices=["random", "human", "random-cmd", "walkthrough"],
                        help="Select an agent to play the game: %(choices)s."
                             " Default: %(default)s.")
    parser.add_argument("--max-steps", type=int, default=0, metavar="STEPS",
                        help="Limit maximum number of steps.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for random and random-cmd agents.")
    parser.add_argument("--state_fn", type=str, default=None,
                        help="Filename for where to store state information for each command.")
    parser.add_argument("--utts_fn", type=str, default=None,
                        help="Filename for where to store utterances (language output).")
    parser.add_argument("--viewer", metavar="PORT", type=int, nargs="?", const=6070,
                        help="Start web viewer.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    parser.add_argument("-vv", "--very-verbose", action="store_true",
                        help="Print debug information.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.very_verbose:
        args.verbose = args.very_verbose
    
    if args.state_fn:
        state_wf = logging.StreamHandler(open(args.state_fn, "w"))
        state_wf.setLevel(logging.INFO)
        state_logger = logging.getLogger("states")
        state_logger.setLevel(logging.INFO)
        state_logger.addHandler(state_wf)
        # open(args.state_fn, 'w')
    if args.utts_fn:
        # bc = logging.basicConfig(filename=args.utts_fn)
        utts_wf = logging.StreamHandler(open(args.utts_fn, "w"))
        # utts_wf = open(args.utts_fn, 'w')
    else:
        # bc = logging.basicConfig()
        utts_wf = logging.StreamHandler(sys.stdout)
    utts_wf.terminator = ''
    utts_wf.setLevel(logging.INFO)
    utts_logger = logging.getLogger("utterances")
    utts_logger.setLevel(logging.INFO)
    utts_logger.addHandler(utts_wf)

    request_infos = textworld.EnvInfos(inventory=True, description=True, facts=True)
    env = textworld.start(args.game, infos=request_infos)

    if args.mode == "random":
        agent = textworld.agents.NaiveAgent(seed=args.seed)
    elif args.mode == "random-cmd":
        # TODO get rid of redundancy
        agent = textworld.agents.RandomCommandAgent(seed=args.seed)
    elif args.mode == "human":
        agent = textworld.agents.HumanAgent()
    elif args.mode == 'walkthrough':
        agent = textworld.agents.WalkthroughAgent()

    agent.reset(env)
    if args.viewer is not None:
        from textworld.envs.wrappers import HtmlViewer
        env = HtmlViewer(env, port=args.viewer)

    if args.mode == "human" or args.very_verbose:
        utts_logger.info("Using {}.\n".format(env.__class__.__name__))

    game_state = env.reset()
    if args.mode == "human" or args.verbose:
        contents = env.render(mode="text")
        contents = '\n'.join(contents.split('\n')[22:])  # remove `TEXTWORLD` heading
        contents = contents.replace('\n\n', '\n')
        utts_logger.info(contents)
        if args.state_fn:
            state_logger.info(json.dumps({
                'facts': [Proposition.serialize(prop) for prop in env.state.facts],
                'inventory': env.state.inventory.replace('\n\n', '\n'),
                'description': env.state.description.replace('\n\n', '\n'),
            }))

    reward = 0
    done = False

    for _ in range(args.max_steps) if args.max_steps > 0 else itertools.count():
        # tw_inform7.py
        command = agent.act(game_state, reward, done)
        # env.state.game.world.facts
        game_state, reward, done = env.step(command)

        if args.mode == "human" or args.verbose:
            contents = env.render(mode="text")
            contents = contents.replace('\n\n', '\n').replace('\n\n', '\n')
            utts_logger.info(contents)
            if args.state_fn:
                state_logger.info(json.dumps({
                    'facts': [Proposition.serialize(prop) for prop in env.state.facts],
                    'inventory': env.state.inventory.replace('\n\n', '\n'),
                    'description': env.state.description.replace('\n\n', '\n'),
                }))

        if done:
            break

    env.close()
    utts_logger.info("Done after {} steps. Score {}/{}.".format(game_state.moves, game_state.score, game_state.max_score))


if __name__ == "__main__":
    main()
