package com.bdilab.httpdexperiment.utils;


import java.io.*;


public class SSHUtils {

    static Boolean flag =true;
    public static String  runSSH(Integer StartServers,Integer MinSpareServers,Integer MaxSpareServers,Integer MaxRequestWorkers,Integer MaxRequestsPerChild) {

        String reuslt = "";
        synchronized(flag){
            if(flag){
                flag=false;

                File file =new File("httpd-mpm.conf");
                try{
                    if(!file.exists()){
                        file.createNewFile();
                    }else{
                        FileWriter filedelete =new FileWriter(file);
                        filedelete.write("");
                        filedelete.flush();
                        filedelete.close();
                    }
                    FileWriter fileWriter = new FileWriter(file);
                    fileWriter.write("<IfModule !mpm_netware_module>\n" +
                            "    PidFile \"logs/httpd.pid\"\n" +
                            "</IfModule>\n" +
                            "<IfModule mpm_prefork_module>\n" +
                            "    StartServers             "+StartServers+"\n" +
                            "    MinSpareServers          "+MinSpareServers+"\n" +
                            "    MaxSpareServers          "+MaxSpareServers+"\n" +
                            "    MaxRequestWorkers        "+MaxRequestWorkers+"\n" +
                            "    MaxConnectionsPerChild   "+MaxRequestsPerChild+"\n" +
                            "</IfModule>\n" +
                            "<IfModule mpm_worker_module>\n" +
                            "    StartServers             3\n" +
                            "    MinSpareThreads         75\n" +
                            "    MaxSpareThreads        250 \n" +
                            "    ThreadsPerChild         25\n" +
                            "    MaxRequestWorkers      400\n" +
                            "    MaxConnectionsPerChild   0\n" +
                            "</IfModule>\n" +
                            "<IfModule mpm_event_module>\n" +
                            "    StartServers             3\n" +
                            "    MinSpareThreads         75\n" +
                            "    MaxSpareThreads        250\n" +
                            "    ThreadsPerChild         25\n" +
                            "    MaxRequestWorkers      400\n" +
                            "    MaxConnectionsPerChild   0\n" +
                            "</IfModule>\n" +
                            "<IfModule mpm_netware_module>\n" +
                            "    ThreadStackSize      65536\n" +
                            "    StartThreads           250\n" +
                            "    MinSpareThreads         25\n" +
                            "    MaxSpareThreads        250\n" +
                            "    MaxThreads            1000\n" +
                            "    MaxConnectionsPerChild   0\n" +
                            "</IfModule>\n" +
                            "<IfModule mpm_mpmt_os2_module>\n" +
                            "    StartServers             2\n" +
                            "    MinSpareThreads          5\n" +
                            "    MaxSpareThreads         10\n" +
                            "    MaxConnectionsPerChild   0\n" +
                            "</IfModule>\n" +
                            "<IfModule mpm_winnt_module>\n" +
                            "    ThreadsPerChild        150\n" +
                            "    MaxConnectionsPerChild   0\n" +
                            "</IfModule>\n" +
                            "<IfModule !mpm_netware_module>\n" +
                            "    MaxMemFree            2048\n" +
                            "</IfModule>\n" +
                            "<IfModule mpm_netware_module>\n" +
                            "    MaxMemFree             100\n" +
                            "</IfModule>");
                    fileWriter.close();
                }catch(IOException e){
                    e.printStackTrace();
                }

                Process procPre;
                Process proc;
                try {
                    System.out.println("apachectl restart");
                    procPre = Runtime.getRuntime().exec("apachectl restart",null);
                    System.out.println(procPre .waitFor());

                    String cmd="ab -c 10000 -t 10  http://127.0.0.1/";
                    System.out.println(cmd);
                    proc = Runtime.getRuntime().exec(cmd,null);
                    BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream(),"utf-8"));
                    String line = null;
                    while ((line = in.readLine()) != null) {
                        System.out.println(line);
                        if(line.contains("Requests per second")){
                            String[] meta = line.split(" ");
                            for(int i=0;i<meta.length;i++){
                                System.out.println(meta[i]+"第"+i+"个");
                            }
                            reuslt =line.split(" ")[6];
                        }
                    }
                    in.close();
                    System.out.println(proc.waitFor());
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                flag=true;
                return reuslt;
            }else{
                return "有参数性能测试正在运行,请稍后再试";
            }
        }
    }
}
