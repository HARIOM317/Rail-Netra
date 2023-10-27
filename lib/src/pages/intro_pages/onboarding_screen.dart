import 'package:flutter/material.dart';
import 'package:railway/src/pages/intro_pages/intro_screen1.dart';
import 'package:railway/src/pages/intro_pages/intro_screen2.dart';
import 'package:railway/src/pages/intro_pages/intro_screen3.dart';
import 'package:railway/src/pages/intro_pages/intro_screen4.dart';
import 'package:railway/src/pages/intro_pages/intro_screen5.dart';
import 'package:railway/src/pages/splash_screens/login_welcome_splash_screen.dart';
import 'package:railway/src/utils/constants.dart';
import 'package:smooth_page_indicator/smooth_page_indicator.dart';

class IntroScreen extends StatefulWidget {
  const IntroScreen({super.key});

  @override
  State<IntroScreen> createState() => _IntroScreenState();
}

class _IntroScreenState extends State<IntroScreen> {
  // controller to keep track of which page we are on
  final PageController _controller = PageController();

  // keep track of if we are on the last page or not
  bool onLastPage = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // page view
          PageView(
            controller: _controller,
            onPageChanged: (index) {
              setState(() {
                onLastPage = (index == 4);
              });
            },
            children: const [
              IntroPage1(),
              IntroPage2(),
              IntroPage3(),
              IntroPage4(),
              IntroPage5(),
            ],
          ),

          // dot indicator
          Container(
              alignment: const Alignment(0, 0.9),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // skip button
                  GestureDetector(
                      onTap: () {
                        _controller.jumpToPage(4); // indexing start from 0
                      },
                      child: Card(
                        color: const Color(0xfffffcfc),
                        elevation: 2,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                          child: Text('Skip', style: TextStyle(
                            color: primaryColor,
                            fontSize: 18,
                            fontWeight: FontWeight.bold
                          ),
                          ),
                        ),
                      )
                  ),

                  SmoothPageIndicator(
                      controller: _controller,
                      count: 5,
                    axisDirection: Axis.horizontal,

                    effect: ExpandingDotsEffect(
                      activeDotColor: primaryColor,
                      dotColor: primaryColor.withOpacity(0.4),
                      dotHeight: 8,
                      dotWidth: 8
                    ),
                  ),

                  // next button
                  onLastPage
                      ? GestureDetector(
                          onTap: () {
                            Navigator.push(context, MaterialPageRoute(builder: (context) => const LoginWelcomeSplashScreen()));
                          },
                          child: Card(
                            color: const Color(0xfffffcfc),
                            elevation: 2,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                              child: Text("Let's go", style: TextStyle(
                                  color: primaryColor,
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold
                              ),
                              ),
                            ),
                          )
                  )
                      : GestureDetector(
                          onTap: () {
                            _controller.nextPage(
                                duration: const Duration(milliseconds: 200),
                                curve: Curves.linear
                            );
                          },
                          child: Card(
                            color: const Color(0xfffffcfc),
                            elevation: 2,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                              child: Text("Next", style: TextStyle(
                                  color: primaryColor,
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold
                              ),
                              ),
                            ),
                          )
                  ),
                ],
              )),
        ],
      ),
    );
  }
}
