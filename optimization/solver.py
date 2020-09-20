from ortools.linear_solver import pywraplp


def solve(player_scores, player_costs, max_price):
    solver = pywraplp.Solver.CreateSolver('assignment_mip', 'CBC')

    num_players = len(player_scores)
    num_position_types = len(player_scores[0])

    x = {}
    for i in range(num_players):
        for j in range(num_position_types):
            x[i, j] = solver.IntVar(0, 1, '')

    # Constraints:
    for i in range(num_players):
        solver.Add(solver.Sum([x[i, j] for j in range(num_position_types)]) <= 1)

    # 2 goalkeepers
    solver.Add(solver.Sum([x[i, 0] for i in range(num_players)]) == 2)

    # 5 defenders
    solver.Add(solver.Sum([x[i, 1] for i in range(num_players)]) == 5)

    # 5 midfielders
    solver.Add(solver.Sum([x[i, 2] for i in range(num_players)]) == 5)

    # 3 forwards
    solver.Add(solver.Sum([x[i, 3] for i in range(num_players)]) == 3)

    # Cost constraint
    solver.Add(
        solver.Sum(
            [sum([x[i, j] * player_costs[i] for i in range(num_players)])
             for j in range(num_position_types)]) <= max_price)

    objective_terms = []
    for i in range(num_players):
        for j in range(num_position_types):
            objective_terms.append(player_scores[i][j] * x[i, j])

    solver.Maximize(solver.Sum(objective_terms))
    status = solver.Solve()

    team = {}
    if status == pywraplp.Solver.OPTIMAL:
        print('Total score = ', solver.Objective().Value(), "\n")
        for i in range(num_players):
            for j in range(num_position_types):
                if x[i, j].solution_value() > 0.5:
                    team[i] = (player_scores[i][j], j)

    return team


def create_score_matrix(roles, scores):
    """
    Given a mapping from team member to role and a mapping from team member to
    score, creates a sparse score matrix with varying rows representing teammates
    and varying columns representing the different positions. Each cell would be
    filled with a score.
    :param roles:
    :param scores:
    :return:
    """
    possible_roles = list(set(roles.values()))
    player_order = []
    score_matrix = []
    for (person, score) in scores.items():
        row = [0 for i in range(len(possible_roles))]
        row[possible_roles.index(roles[person])] = score
        player_order.append(person)
        score_matrix.append(row)

    return score_matrix, possible_roles, player_order


def choose_gameweek_team(roster, predicted_score):

    score_matrix, role_order, player_order = create_score_matrix(roster, predicted_score)
    choices = optimize_gameweek_team(score_matrix, role_order)

    team = {}
    for pick in choices:
        team[player_order[pick]] = choices[pick]

    return team


def optimize_gameweek_team(score_matrix, role_order):
    """
    Given a mapping from roster member to predicted score, returns the team of
    eleven that should be used for scoring. A team contains eleven members, including
    a captain, and a co-captain.

    The following restriction applies to choosing a team for a gameweek:\n
    * must have exactly one goalkeeper \n
    * must have at least 3 defenders \n
    * must have at least 1 forwards
    :param roster: A mapping of 15 players and their positions
    :param predicted_score: A mapping from the 15 players to their potential scores
    :return: A list of eleven players, where the first and second players are captain
            and co-captain, respectively.
    """

    solver = pywraplp.Solver.CreateSolver('assignment_mip', 'CBC')
    needed_players = 11
    num_players = len(score_matrix)
    num_position_types = len(score_matrix[0])

    x = {}
    for i in range(num_players):
        for j in range(num_position_types):
            x[i, j] = solver.IntVar(0, 1, '')

    for i in range(num_players):
        solver.Add(solver.Sum([x[i, j] for j in range(num_position_types)]) <= 1)

    # Exactly 1 goalkeeper
    solver.Add(solver.Sum([x[i, role_order.index('Goalkeeper')] for i in range(num_players)]) == 1)

    # At least 3 defenders
    solver.Add(solver.Sum([x[i, role_order.index('Defender')] for i in range(num_players)]) >= 3)

    # At least 1 forward
    solver.Add(solver.Sum([x[i, role_order.index('Forward')] for i in range(num_players)]) >= 1)

    # Must be exactly 11 players
    solver.Add(
        solver.Sum(
            [sum([x[i, j] for i in range(num_players)])
             for j in range(num_position_types)]) == needed_players)

    objective_terms = []
    for i in range(num_players):
        for j in range(num_position_types):
            objective_terms.append(score_matrix[i][j] * x[i, j])

    solver.Maximize(solver.Sum(objective_terms))
    status = solver.Solve()

    team = {}
    if status == pywraplp.Solver.OPTIMAL:
        print('Total score =', solver.Objective().Value(), "\n")
        for i in range(num_players):
            for j in range(num_position_types):
                if x[i, j].solution_value() > 0.5:
                    team[i] = (score_matrix[i][j], j)

    return team


def swap_goalie():
    pass


def swap_captain(captains):
    temp = captains["captain"]
    captains["captain"] = captains["co-captain"]
    captains["co-captain"] = temp


def swap_players(current_players, new_data):
    """
    Given data for a new week, find the optimal score for swapping.
    :param current_players:
    :param new_data:
    :return:
    """
    return current_players, new_data
