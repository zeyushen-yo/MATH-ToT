def get_task(name):
    if name == 'MATH2':
        from tot.tasks.MATH2 import Math2Task
        return Math2Task()        
    else:
        raise NotImplementedError