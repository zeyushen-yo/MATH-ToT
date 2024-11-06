def get_task(name):
    if name == 'MATH':
        from tot.tasks.MATH import MathTask
        return MathTask()
    else:
        raise NotImplementedError