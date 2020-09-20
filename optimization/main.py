import solver
import csv


POSITION_MAP = {
    'Goalkeeper': 0,
    'Defender': 1,
    'Midfielder': 2,
    'Forward': 3
}

INV_POSITION_MAP = {
    0: 'Goalkeeper',
    1: 'Defender',
    2: 'Midfielder',
    3: 'Forward'
}


def main():
    max_cost = 1000
    player_scores, player_costs, all_player_names = read_in_data_csv("2018-2019_data.csv")

    print("====== ROSTER ======")
    team = solver.solve(player_scores, player_costs, max_cost)
    for member in team:
        print(all_player_names[member], team[member])

    roles = {}
    scores = {}
    for key in team:
        scores[key] = team[key][0]
        roles[key] = INV_POSITION_MAP[team[key][1]]

    print()
    print("====== Team to Play ======")
    team = solver.choose_gameweek_team(roles, scores)
    for member in team:
        print(all_player_names[member], team[member])


def read_in_data_csv(filename):
    file = open(filename, encoding='utf-8')
    csv_reader = csv.reader(file)


    cols = next(csv_reader)
    pts_col = cols.index("total_points")
    cost_col = cols.index("now_cost")
    position_col = cols.index("position")
    last_name_col = cols.index("second_name")
    first_name_col = cols.index("first_name")

    player_scores = []
    player_names = []
    player_costs = []

    for row in csv_reader:
        player_score = [0, 0, 0, 0]
        player_score[POSITION_MAP[row[position_col]]] += int(row[pts_col])
        player_scores.append(player_score)
        player_costs.append(int(row[cost_col]))
        player_names.append(row[first_name_col] + " " + row[last_name_col])

    return player_scores, player_costs, player_names


if __name__ == "__main__":
    main()
