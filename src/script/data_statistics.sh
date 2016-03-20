#!/bin/bash

echo "the total number of songs:"
wc -l ../../data/mars_tianchi_songs.csv

echo "the total number of singers:"
awk -F ',' '{print $2}' ../../data/mars_tianchi_songs.csv | sort | uniq | wc -l

echo "the total number of actions:"
wc -l ../../data/mars_tianchi_user_actions.csv

echo "the total number of users:"
awk -F ',' '{print $1}' ../../data/mars_tianchi_user_actions.csv | sort | uniq | wc -l

echo "the number of action for each type(1: play 2: download 3:collection):"
awk -F ',' '{print $4}' ../../data/mars_tianchi_user_actions.csv | sort | uniq -c

