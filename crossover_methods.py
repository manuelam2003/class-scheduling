import numpy as np
import random
from helpers import repair_balance
from parameters import n, affinity_matrix

# Crossover: Single-point crossover
def single_point_crossover(parent1, parent2):
    point = random.randint(1, n - 1)
    child1 = np.vstack((parent1[:point], parent2[point:]))
    child2 = np.vstack((parent2[:point], parent1[point:]))
    return repair_balance(child1), repair_balance(child2)

def uniform_crossover(parent1, parent2):
    mask = np.random.randint(0, 2, parent1.shape, dtype=bool)  # Random binary mask
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return repair_balance(child1), repair_balance(child2)

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(n), 2))
    child1 = np.vstack((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    child2 = np.vstack((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    return repair_balance(child1), repair_balance(child2)

def heuristic_crossover(parent1, parent2):
    """
    Perform heuristic crossover using the affinity matrix.
    Each student is assigned to the group from the parent with higher affinity.
    """
    child = np.zeros_like(parent1, dtype=int)  # Initialize empty child chromosome

    for student in range(n):
        # Get the groups assigned in both parents
        group_parent1 = np.where(parent1[student] == 1)[0][0]
        group_parent2 = np.where(parent2[student] == 1)[0][0]

        # Compare affinity values
        affinity1 = affinity_matrix[student, group_parent1]
        affinity2 = affinity_matrix[student, group_parent2]

        # Assign the group from the parent with higher affinity
        if affinity1 > affinity2:
            child[student, group_parent1] = 1
        elif affinity2 > affinity1:
            child[student, group_parent2] = 1
        else:
            # If affinity values are equal, randomly pick one parent's assignment
            chosen_group = random.choice([group_parent1, group_parent2])
            child[student, chosen_group] = 1

    # Ensure the child respects the group size constraints
    child = repair_balance(child)

    # Return two identical children (to match the algorithm structure)
    return child, child  # Both children are the same

def heuristic_crossover2(parent1, parent2, k=37):
    """
    Perform heuristic crossover using the affinity matrix.
    Students with lower affinity in a group are reassigned to better-matching groups.

    Args:
        parent1: The first parent chromosome.
        parent2: The second parent chromosome.
        affinity_matrix: Matrix where entry (i, j) represents the affinity of student i for group j.
        k: Number of students with the lowest affinity to consider for reassignment.

    Returns:
        Two identical children after crossover.
    """
    n, m = affinity_matrix.shape  # Number of students and groups
    child = np.zeros_like(parent1, dtype=int)  # Initialize the child chromosome

    assigned_students = set()  # Track students already assigned to avoid duplicates

    for group in range(m):
        # Step 1: Calculate average affinity for this group in both parents
        avg_affinity_parent1 = np.mean([affinity_matrix[student, group] for student in range(n) if parent1[student, group] == 1])
        avg_affinity_parent2 = np.mean([affinity_matrix[student, group] for student in range(n) if parent2[student, group] == 1])

        # Step 2: Select the parent group with the lower affinity
        low_affinity_parent = parent1 if avg_affinity_parent1 < avg_affinity_parent2 else parent2
        high_affinity_parent = parent2 if avg_affinity_parent1 < avg_affinity_parent2 else parent1

        # Step 3: Identify k students with the lowest affinity in the low-affinity parent group
        group_students = [student for student in range(n) if low_affinity_parent[student, group] == 1]
        student_affinities = [(student, affinity_matrix[student, group]) for student in group_students]
        student_affinities.sort(key=lambda x: x[1])  # Sort by affinity (ascending)
        low_affinity_students = [student for student, _ in student_affinities[:k]]

        for student in low_affinity_students:
            # Step 4: Find a better matching group for this student in the high-affinity parent
            best_group = np.argmax(affinity_matrix[student, :])  # Group with the highest affinity for this student
            if best_group != group and student not in assigned_students:
                # Assign the student to the better group
                child[student, best_group] = 1
                assigned_students.add(student)

        # Step 5: Assign remaining students in the current group from the low-affinity parent
        for student in group_students:
            if student not in assigned_students:
                child[student, group] = 1
                assigned_students.add(student)

    # Step 6: Assign remaining unassigned students randomly
    for student in range(n):
        if student not in assigned_students:
            random_group = random.randint(0, m - 1)  # Randomly pick a group
            child[student, random_group] = 1

    # Ensure the child respects the group size constraints
    child = repair_balance(child)

    # Return two identical children (to match the algorithm structure)
    return child, child

def stochastic_heuristic_crossover(parent1, parent2, k=8, randomness=0.2):
    """
    Perform stochastic heuristic crossover using the affinity matrix.
    Introduces randomness in decision-making to increase diversity.

    Args:
        parent1: The first parent chromosome.
        parent2: The second parent chromosome.
        k: Number of students with the lowest affinity to consider for reassignment.
        randomness: Probability of making a random choice instead of the heuristic decision.

    Returns:
        Two identical children after crossover.
    """
    n, m = affinity_matrix.shape  # Number of students and groups
    child = np.zeros_like(parent1, dtype=int)  # Initialize the child chromosome

    assigned_students = set()  # Track students already assigned to avoid duplicates

    for group in range(m):
        # Step 1: Calculate average affinity for this group in both parents
        avg_affinity_parent1 = np.mean([affinity_matrix[student, group] for student in range(n) if parent1[student, group] == 1])
        avg_affinity_parent2 = np.mean([affinity_matrix[student, group] for student in range(n) if parent2[student, group] == 1])

        # Step 2: Stochastically decide whether to use high or low affinity group
        if random.random() < randomness:
            # Randomly pick one parent's group assignments
            selected_parent = random.choice([parent1, parent2])
        else:
            # Select the low-affinity group deterministically
            selected_parent = parent1 if avg_affinity_parent1 < avg_affinity_parent2 else parent2

        # Step 3: Identify k students with the lowest affinity in the selected group
        group_students = [student for student in range(n) if selected_parent[student, group] == 1]
        student_affinities = [(student, affinity_matrix[student, group]) for student in group_students]
        student_affinities.sort(key=lambda x: x[1])  # Sort by affinity (ascending)
        low_affinity_students = [student for student, _ in student_affinities[:k]]

        for student in low_affinity_students:
            # Step 4: Stochastically pick a better matching group or the current group
            if random.random() < randomness:
                # Randomly assign the student to any valid group
                random_group = random.choice(range(m))
                child[student, random_group] = 1
                assigned_students.add(student)
            else:
                # Assign the student to the group with the highest affinity
                best_group = np.argmax(affinity_matrix[student, :])
                if best_group != group and student not in assigned_students:
                    child[student, best_group] = 1
                    assigned_students.add(student)

        # Step 5: Assign remaining students in the current group deterministically
        for student in group_students:
            if student not in assigned_students:
                child[student, group] = 1
                assigned_students.add(student)

    # Step 6: Assign remaining unassigned students randomly
    for student in range(n):
        if student not in assigned_students:
            random_group = random.randint(0, m - 1)  # Randomly pick a group
            child[student, random_group] = 1

    # Ensure the child respects the group size constraints
    child = repair_balance(child)

    # Return two identical children (to match the algorithm structure)
    return child, child
