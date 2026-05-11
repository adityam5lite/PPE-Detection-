violation_count = 0

def update_analytics(violations):

    global violation_count

    violation_count += len(violations)

    print("Total Violations:", violation_count)