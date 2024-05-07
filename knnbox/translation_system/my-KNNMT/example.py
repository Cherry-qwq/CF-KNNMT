def launch_app(args, app, parentpid=-1):
    def start_app():
        try:
            app.serve_forever()
        except:
            exit(0)
    
    def stop_app(signum, f):
        print("Stop!")
        app.socket.close()
        exit(0)
    
    signal.signal(signal.SIGINT, stop_app)
    
    if parentpid >= 0:
        print(f"Started documents server in subprocess {os.getpid()} on http://localhost:{args.port}  |  Manager process id = {parentpid}")
    else:
        print(f"Started documents server on http://localhost:{args.port}")
        
    print("Remote access to the documents will be " + ("rejected." if args.reject_remote_access else "accepted."))
    
    server_thread = threading.Thread(target=start_app, args=[])
    server_thread.run()
    server_thread.join()
    
if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument("--port", default=12345, type=int, help='Local port for the service, default to 12345.')
    ps.add_argument("--reject-remote-access", default=False, action='store_true',
                    help='Reject accessing requests from other computers in your LAN. By default the document could be accessed by others in your LAN, turn on this option to reject them.')
    ps.add_argument("--do-not-fork", default=False, action='store_true', help='Single process mode. If you are debugging Documents Server, remember to add this argument to make sure the debugger could trace the server.')
    ps.add_argument("--run-in-subprocess", default=None, type=int, help='Internel use, do not care this argument.')
    args = ps.parse_args()
 

    if args.do_not_fork or ('win' in sys.platform):
        app = init_app(args)
        launch_app(args, app)
    else:
        if not args.run_in_subprocess is None:
            app = init_app(args)
            launch_app(args, app, args.run_in_subprocess) 
        else:
            pid = os.getpid()
            p = subprocess.Popen([sys.executable] + sys.argv + ["--run-in-subprocess", str(pid)])   
            
            def stop_subprocess(sig = signal.SIGINT):
                p.send_signal(signal.SIGINT)
                p.wait()
                    
            recursive_guard = 0
            def sig_stop_shell(signum, f):
                global recursive_guard
                if recursive_guard == 0:
                    recursive_guard = 1
                    stop_subprocess()
                   
                exit(0)
                
            def sig_subprocess_exit(signum, f):
                global recursive_guard, p
                if recursive_guard == 0:
                    print('''The subprocess early exited probably because of an exception. 
        Manager process will try to relaunch it in 5s, enter command 'exit' or type Ctrl-C to force exit.''')
                    time.sleep(5)
                    p = subprocess.Popen([sys.executable] + sys.argv + ["--run-in-subprocess", str(pid)])
                else:
                    exit(0)
                   
            signal.signal(signal.SIGINT, sig_stop_shell)
            signal.signal(signal.SIGCHLD, sig_subprocess_exit)
            
            help_msg = '''Manager process shell
    Available commands:
        help        : Show this message
        exit        : Stop documents server
        restart     : Restart documents server        
>>>'''
            time.sleep(2)
            print(help_msg, end='')
            first_prompt = True
            while True:
                if not first_prompt:
                    print(">>>", end='')
                s = input().strip()
                if s == 'exit':
                    sig_stop_shell(0,0)
                elif s == 'restart':
                    stop_subprocess()
                elif s == 'help':
                    print(help_msg)
                elif len(s) != 0:
                    print("Unkown command. Available commands: help exit restart")
                    
                first_prompt = False