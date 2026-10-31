using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

// Creates the launcher suspended, assigns it to a private job before it can
// create descendants, and can terminate only that job's processes.
public sealed class EditorOwnedJob : IDisposable {
    const uint CREATE_SUSPENDED = 4, STARTF_USESTDHANDLES = 0x100, INHERIT = 1;
    const uint WAIT_OBJECT_0 = 0, WAIT_TIMEOUT = 258, STILL_ACTIVE = 259;
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    struct Startup { public uint cb; public string reserved, desktop, title; public uint x,y,xsize,ysize,xchars,ychars,fill,flags; public ushort show,reserved2; public IntPtr reservedPtr, stdin, stdout, stderr; }
    [StructLayout(LayoutKind.Sequential)] struct ProcessInfo { public IntPtr process, thread; public uint pid, tid; }
    [StructLayout(LayoutKind.Sequential)] struct Accounting { public long totalUser, totalKernel, periodUser, periodKernel; public uint faults, total, active, terminated; }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool CreateProcess(string app, StringBuilder command, IntPtr pa, IntPtr ta, bool inherit, uint flags, IntPtr env, string cwd, ref Startup startup, out ProcessInfo info);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)] static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError=true)] static extern uint WaitForSingleObject(IntPtr handle, uint ms);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr handle, out uint code);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool QueryInformationJobObject(IntPtr job, int cls, out Accounting data, uint length, IntPtr returned);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateJobObject(IntPtr job, uint code);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateProcess(IntPtr process, uint code);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    IntPtr job, process; public uint Pid {get; private set;}
    static Exception Error(string where) { return new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), where); }
    static string Quote(string s) {
        var b = new StringBuilder("\""); int slash = 0;
        foreach (char c in s) { if (c == '\\') { slash++; continue; } if (c == '"') { b.Append('\\', slash*2+1); b.Append(c); slash=0; continue; } b.Append('\\',slash); slash=0; b.Append(c); }
        b.Append('\\', slash*2); b.Append('"'); return b.ToString();
    }
    public EditorOwnedJob(string exe, string[] args, string stdout, string stderr) {
        job = CreateJobObject(IntPtr.Zero,null); if(job == IntPtr.Zero) throw Error("CreateJobObject");
        ProcessInfo pi = new ProcessInfo();
        try {
            using(var outFile = new FileStream(stdout,FileMode.CreateNew,FileAccess.Write,FileShare.ReadWrite))
            using(var errFile = new FileStream(stderr,FileMode.CreateNew,FileAccess.Write,FileShare.ReadWrite))
            using(var input = new FileStream("NUL",FileMode.Open,FileAccess.Read,FileShare.ReadWrite)) {
                var oh=outFile.SafeFileHandle.DangerousGetHandle(); var eh=errFile.SafeFileHandle.DangerousGetHandle(); var ih=input.SafeFileHandle.DangerousGetHandle();
                if(!SetHandleInformation(oh,INHERIT,INHERIT) || !SetHandleInformation(eh,INHERIT,INHERIT) || !SetHandleInformation(ih,INHERIT,INHERIT)) throw Error("SetHandleInformation");
                var si=new Startup(); si.cb=(uint)Marshal.SizeOf(typeof(Startup)); si.flags=STARTF_USESTDHANDLES; si.stdin=ih; si.stdout=oh; si.stderr=eh;
                var command=new StringBuilder(Quote(exe)); foreach(var arg in args) command.Append(' ').Append(Quote(arg));
                if(!CreateProcess(exe,command,IntPtr.Zero,IntPtr.Zero,true,CREATE_SUSPENDED,IntPtr.Zero,Path.GetDirectoryName(exe),ref si,out pi)) throw Error("CreateProcess");
                if(!AssignProcessToJobObject(job,pi.process)) { TerminateProcess(pi.process,1); throw Error("AssignProcessToJobObject"); }
                process=pi.process; Pid=pi.pid;
                if(ResumeThread(pi.thread)==uint.MaxValue) throw Error("ResumeThread");
                CloseHandle(pi.thread); pi.thread=IntPtr.Zero;
            }
        } catch { if(process!=IntPtr.Zero) TerminateJobObject(job,1); if(pi.thread!=IntPtr.Zero) CloseHandle(pi.thread); if(pi.process!=IntPtr.Zero && process==IntPtr.Zero) CloseHandle(pi.process); Dispose(); throw; }
    }
    public bool Wait(int ms) { uint r=WaitForSingleObject(process,(uint)ms); if(r==WAIT_OBJECT_0) return true; if(r==WAIT_TIMEOUT) return false; throw Error("WaitForSingleObject"); }
    public int ExitCode { get { uint n; if(!GetExitCodeProcess(process,out n)) throw Error("GetExitCodeProcess"); if(n==STILL_ACTIVE) throw new InvalidOperationException("still running"); return unchecked((int)n); } }
    public int Active { get { Accounting data; if(!QueryInformationJobObject(job,1,out data,(uint)Marshal.SizeOf(typeof(Accounting)),IntPtr.Zero)) throw Error("QueryInformationJobObject"); return (int)data.active; } }
    public void TerminateAndWait(int milliseconds) {
        if(Active != 0 && !TerminateJobObject(job,1)) throw Error("TerminateJobObject");
        var deadline=DateTime.UtcNow.AddMilliseconds(milliseconds);
        while(Active != 0 && DateTime.UtcNow < deadline) Thread.Sleep(50);
        if(Active != 0) throw new TimeoutException("Owned job processes did not exit");
    }
    public void Dispose() { if(process!=IntPtr.Zero) { CloseHandle(process); process=IntPtr.Zero; } if(job!=IntPtr.Zero) { CloseHandle(job); job=IntPtr.Zero; } }
}
