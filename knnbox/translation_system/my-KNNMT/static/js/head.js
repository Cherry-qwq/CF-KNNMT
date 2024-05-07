document.writeln("<!DOCTYPE html>");
document.writeln("<html>");
document.writeln("    ");
document.writeln("    <head>");
document.writeln("        <title>Transportation Game</title>");
document.writeln("        <meta charset=\'utf-8\'>");
document.writeln("        <meta http-equiv=\'X-UA-Compatible\' content=\'IE=edge,chrome=1\'>");
document.writeln("        <script src=\'static/js/jquery-3.6.3.min.js\'></script>");
document.writeln("        <script src=\'static/js/semantic.js\'></script>");
document.writeln("        <link rel=\'stylesheet\' type=\'text/css\' href=\'static/css/semantic.css\'>");
document.writeln("");
document.writeln("        {% if is_mobile %} ");
document.writeln("        <meta name=\'viewport\' content=\'width=device-width,height=device-height,initial-scale=1\'> ");
document.writeln("        {% endif %}");
//document.writeln("        <script type=\'text/javascript\' src=\'head.js\'></script>");
document.writeln("    </head>");
document.writeln("    <body>");
document.writeln("");
document.writeln("        <!-- 上方的MenuBar 开始 -->");
document.writeln("        <div class=\'ui top horizental inverted huge sidebar menu visible\' id=\'menubar\'> ");
document.writeln("            <div class=\'item\'>Logo</div>    ");
document.writeln("            <!-- -->");
document.writeln("            <div style=\'direction: rtl; width:100%\'>");
document.writeln("                <div style=\'width:fit-content; display:flex; direction:ltr\'>");
document.writeln("                    <a class=\'item\' onmousemove=\'MenuBarItemOver(this)\'; onmouseout=\'MenuBarItemOut(this)\'");
document.writeln("                        href=\'/\'>");
document.writeln("                        主页");
document.writeln("                    </a>");
document.writeln("                    <a class=\'item\' onmousemove=\'MenuBarItemOver(this)\'; onmouseout=\'MenuBarItemOut(this)\'");
document.writeln("                        href=\'/rank\'>");
document.writeln("                        排行榜");
document.writeln("                    </a>");
document.writeln("                    <a class=\'item\' onmousemove=\'MenuBarItemOver(this)\'; onmouseout=\'MenuBarItemOut(this)\'");
document.writeln("                        onclick=\'login_register()\'>");
document.writeln("                        登录");
document.writeln("                    </a>");
document.writeln("                    <a class=\'item\' onmousemove=\'MenuBarItemOver(this)\'; onmouseout=\'MenuBarItemOut(this)\'");
document.writeln("                        href=\'/upload\'>");
document.writeln("                        上传结果");
document.writeln("                    </a>");
document.writeln("                </div>");
document.writeln("            </div>");
document.writeln("            <script type=\'text/javascript\'>");
document.writeln("                function MenuBarItemOver(obj) {");
document.writeln("                    obj.className = \'active item\';");
document.writeln("                };");
document.writeln("                function MenuBarItemOut(obj) {");
document.writeln("                    obj.className = \'item\';");
document.writeln("                };");
document.writeln("            </script>");
document.writeln("        </div>");
document.writeln("        <!-- 上方的MenuBar 结束 -->");
document.writeln("");
document.writeln("");
document.writeln("        <!-- 登录/注册 modal 开始 -->");
document.writeln("<div id=\'login_register_modal\' class=\'ui modal\'>");
document.writeln("    <div class=\'ui two item stackable tabs menu\'>");
document.writeln("        <a class=\'active item\' data-tab=\'login_tab\' id=\'login_tab_mitem\' onclick=\'activeLoginTab();\'>");
document.writeln("            登录");
document.writeln("        </a>");
document.writeln("        <a class=\'item\' data-tab=\'register_tab\' id=\'register_tab_mitem\' onclick=\'activeRegisterTab();\'>");
document.writeln("            注册");
document.writeln("        </a>");
document.writeln("    </div>");
document.writeln("    <div class=\'ui active tab\' data-tab=\'login_tab\' id=\'login_tab\'>");
document.writeln("        <form class=\'ui large form\' action=\'/login\' method=\'post\'>");
document.writeln("            <div class=\'ui existing segment\'>");
document.writeln("                <!-- username -->");
document.writeln("                <div class=\'field\'>");
document.writeln("                    <div class=\'ui left input\'>");
document.writeln("                        <input id=\'login_username\' name=\'username\' placeholder=\'用户名\' type=\'text\' required>");
document.writeln("                    </div>");
document.writeln("                </div>");
document.writeln("    ");
document.writeln("                <!-- password -->");
document.writeln("                <div class=\'field\'>");
document.writeln("                    <div class=\'ui left input\'>");
document.writeln("                        <input id=\'login_password\' name=\'password\' placeholder=\'密码\' type=\'password\' required>");
document.writeln("                    </div>");
document.writeln("                </div>");
document.writeln("    ");
document.writeln("                <div class=\'field\'>");
document.writeln("                    <button type=\'submit\' class=\'ui fluid primary button\'>登录</button>");
document.writeln("                </div>");
document.writeln("            </div>");
document.writeln("        </form>");
document.writeln("    </div>");
document.writeln("    ");
document.writeln("    <div class=\'ui tab\' data-tab=\'register_tab\' id=\'register_tab\'>");
document.writeln("        <form class=\'ui large form\' action=\'/register\' method=\'post\'>");
document.writeln("            <div class=\'ui existing segment\'>");
document.writeln("                <!-- username -->");
document.writeln("                <div class=\'field\'>");
document.writeln("                    <div class=\'ui left input\'>");
document.writeln("                        <input id=\'register_username\' name=\'username\' placeholder=\'用户名\' type=\'text\' required>");
document.writeln("                    </div>");
document.writeln("                </div>");
document.writeln("    ");
document.writeln("                <!-- password -->");
document.writeln("                <div class=\'field\'>");
document.writeln("                    <div class=\'ui left input\'>");
document.writeln("                        <input id=\'register_password\' name=\'password\' type=\'password\' placeholder=\'密码\' required>");
document.writeln("                    </div>");
document.writeln("                </div>");
document.writeln("    ");
document.writeln("                <!-- confirm password -->");
document.writeln("                <div class=\'field\'>");
document.writeln("                    <div class=\'ui left input\'>");
document.writeln("                        <input id=\'register_password_confirm\' name=\'password_confirm\' type=\'password\' placeholder=\'确认密码\' required>");
document.writeln("                    </div>");
document.writeln("                </div>");
document.writeln("    ");
document.writeln("                <!-- email -->");
document.writeln("                <div class=\'field\'>");
document.writeln("                    <div class=\'ui left input\'>");
document.writeln("                        <input id=\'register_email\' name=\'email\' placeholder=\'电子邮件\' type=\'text\' required>");
document.writeln("                    </div>");
document.writeln("                </div>");
document.writeln("    ");
document.writeln("                <div class=\'field\'>");
document.writeln("                    <button type=\'submit\' class=\'ui fluid primary button\'>注册</button>");
document.writeln("                </div>");
document.writeln("            </div>");
document.writeln("        </form>");
document.writeln("    </div>");
document.writeln("    <script type=\'text/javascript\'>");
document.writeln("        function activeLoginTab() {");
document.writeln("            $(\'#login_tab\').addClass(\'active\');");
document.writeln("            $(\'#login_tab_mitem\').addClass(\'active\');");
document.writeln("            ");
document.writeln("            $(\'#register_tab\').removeClass(\'active\');");
document.writeln("            $(\'#register_tab_mitem\').removeClass(\'active\');");
document.writeln("        };");
document.writeln("        function activeRegisterTab() {");
document.writeln("            $(\'#register_tab\').addClass(\'active\');");
document.writeln("            $(\'#register_tab_mitem\').addClass(\'active\');");
document.writeln("            ");
document.writeln("            $(\'#login_tab\').removeClass(\'active\');");
document.writeln("            $(\'#login_tab_mitem\').removeClass(\'active\');");
document.writeln("        };");
document.writeln("        function login_register() {");
document.writeln("            $(\'#login_register_modal\').modal(\'show\');");
document.writeln("        }");
document.writeln("    </script>");
document.writeln("</div>");
document.writeln("</body>");
document.writeln("</html>");
document.writeln("<!-- 登录/注册 modal 结束 -->");