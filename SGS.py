
def serial_schedule_generation(instance, priority_list):
    """
    Generate a feasible RCPSP schedule using the Serial Schedule
    Generation Scheme (SSGS).

    Instance format:
        {
            "durations": {activity: duration},
            "demands": {activity: [resource demands]},
            "capacities": [resource capacities],
            "predecessors": {activity: [predecessors]}
        }

    Returns:
        start_times, end_times, makespan
    """

    durations = instance["durations"]
    demands = instance["demands"]
    capacities = instance["capacities"]
    predecessors = instance["predecessors"]

    activities = list(durations)
    priority = list(priority_list)

    if len(priority) != len(set(priority)) or set(priority) != set(activities):
        raise ValueError("Priority list must contain every activity exactly once.")

    # Check whether the priority list is precedence feasible
    position = {activity: i for i, activity in enumerate(priority_list)}

    for activity in priority_list:
        for predecessor in instance["predecessors"][activity]:
            if position[predecessor] > position[activity]:
                raise ValueError(
                    f"Priority list is not precedence feasible: "
                    f"activity {activity} appears before predecessor {predecessor}."
                )
    n_resources = len(capacities)

    # Resource usage at each discrete time point
    resource_usage = []

    start_times = {}
    end_times = {}
    unscheduled = set(activities)

    while unscheduled:
        # Select the highest-priority activity whose predecessors
        # have all been scheduled.
        activity = next(
            (a for a in priority
             if a in unscheduled
             and all(p in start_times for p in predecessors[a])),
            None
        )

        if activity is None:
            raise ValueError("Precedence graph contains a cycle or invalid predecessor.")

        duration = durations[activity]

        # Earliest possible start based on precedence constraints
        earliest_start = max(
            (end_times[p] for p in predecessors[activity]),
            default=0
        )

        start = earliest_start

        # Find the earliest time at which all resource constraints hold
        while True:
            feasible = True

            for t in range(start, start + duration):
                for r in range(n_resources):
                    current_usage = (
                        resource_usage[t][r]
                        if t < len(resource_usage)
                        else 0
                    )

                    if current_usage + demands[activity][r] > capacities[r]:
                        feasible = False
                        break

                if not feasible:
                    break

            if feasible:
                break

            start += 1

        finish = start + duration

        # Extend the resource profile if needed
        while len(resource_usage) < finish:
            resource_usage.append([0] * n_resources)

        # Reserve resources over [start, finish)
        for t in range(start, finish):
            for r in range(n_resources):
                resource_usage[t][r] += demands[activity][r]

        start_times[activity] = start
        end_times[activity] = finish
        unscheduled.remove(activity)

    makespan = max(end_times.values(), default=0)

    return start_times, end_times, makespan
