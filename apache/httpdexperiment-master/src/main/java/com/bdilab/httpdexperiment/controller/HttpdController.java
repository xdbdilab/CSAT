package com.bdilab.httpdexperiment.controller;

import com.bdilab.httpdexperiment.utils.SSHUtils;
import io.swagger.annotations.ApiParam;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.io.*;

@Controller
@CrossOrigin
@RequestMapping("/experiment")
public class HttpdController {

    @ResponseBody
    @RequestMapping(value = "/httpd",method = RequestMethod.GET)
    public String httpd(
            @ApiParam(name = "StartServers", value = "启动时开启的子进程数量", type = "int")
            @RequestParam(name = "StartServers") Integer StartServers,
            @ApiParam(name = "MinSpareServers", value = " 最小空闲的子进程数", type = "int")
            @RequestParam(name = "MinSpareServers") Integer MinSpareServers,
            @ApiParam(name = "MaxSpareServers", value = "最大空闲的子进程数", type = "int")
            @RequestParam(name = "MaxSpareServers") Integer MaxSpareServers,
            @ApiParam(name = "MaxRequestWorkers ", value = "最大连接的客户数量，影响并发", type = "int")
            @RequestParam(name = "MaxRequestWorkers") Integer MaxRequestWorkers ,
            @ApiParam(name = "MaxRequestsPerChild", value = "子进程处理多少个请求后自动销毁，默认为0，永不销毁", type = "int")
            @RequestParam(name = "MaxRequestsPerChild") Integer MaxRequestsPerChild

            ){

        return SSHUtils.runSSH(StartServers,MinSpareServers,MaxSpareServers,MaxRequestWorkers,MaxRequestsPerChild);
    }

    @ResponseBody
    @GetMapping()
    @RequestMapping(value = "/hello",method = RequestMethod.GET)
    public String hello( ){
        return "helloworld";
    }
}
