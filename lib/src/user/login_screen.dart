// ignore_for_file: use_build_context_synchronously

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/components/custom_textfield.dart';
import 'package:railway/src/components/primary_button.dart';
import 'package:railway/src/components/secondary_button.dart';
import 'package:railway/src/db/share_pref.dart';
import 'package:railway/src/pages/splash_screens/main_splash_screen.dart';
import 'package:railway/src/user/forgot_password_page.dart';
import 'package:railway/src/user/register_user.dart';
import 'package:railway/src/utils/constants.dart';
import 'package:railway/src/utils/custom_page_route.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  bool isPasswordHide = true;
  final _formKey = GlobalKey<FormState>();
  final _formData = <String, Object>{};
  bool isLoading = false;

  // function to disable back button
  Future<bool> _onPop() async {
    return false;
  }

  // on submit function
  _onSubmit() async {
    _formKey.currentState!.save();

    try {
      setState(() {
        isLoading = true;
      });
      // progressIndicator(context);
      UserCredential userCredential = await FirebaseAuth.instance
          .signInWithEmailAndPassword(
              email: _formData['email'].toString(),
              password: _formData['password'].toString());

      if (userCredential.user != null) {
        setState(() {
          isLoading = false;
        });
        FirebaseFirestore.instance
            .collection('users')
            .doc(userCredential.user!.uid)
            .get()
            .then((value) {
          if (value['type'] == 'user') {
            MySharedPreference.saveUserType('user');
            Navigator.push(context, MaterialPageRoute(builder: (context) => const MainSplashScreen()));
          }
        });
        // MySharedPreference.saveUser();
        if (!context.mounted) return;
        Navigator.push(context, MaterialPageRoute(builder: (context) => const MainSplashScreen()));
      }
    } on FirebaseAuthException catch (e) {
      setState(() {
        isLoading = false;
      });

      if (e.code == 'user-not-found') {
        showAlertDialogueBox(context, "No user found for that email.");
      } else if (e.code == 'wrong-password') {
        showAlertDialogueBox(context, "Invalid username or password!");
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: _onPop,

      child: Scaffold(
        body: Container(
          width: double.infinity,
          height: double.infinity,
          decoration: BoxDecoration(
            color: bgColor
          ),
          child: Center(
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Stack(
                  children: [
                    isLoading
                        ? progressIndicator(context)
                        : SingleChildScrollView(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              crossAxisAlignment: CrossAxisAlignment.center,
                              children: [
                                // app icon
                                Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Image.asset(
                                    "assets/images/icons/logo.png",
                                    width: 120,
                                  ),
                                ),

                                // login text
                                Padding(
                                  padding: const EdgeInsets.all(15.0),
                                  child: Text(
                                    "Welcome",
                                    style: TextStyle(
                                      fontSize: 30,
                                      color: primaryColor,
                                      fontWeight: FontWeight.w500
                                    ),
                                  ),
                                ),

                                // form
                                Form(
                                  key: _formKey,
                                  child: Column(
                                    children: [
                                      Padding(
                                        padding: const EdgeInsets.all(8.0),
                                        child: LoginTextField(
                                          hintText: "Email",
                                          textInputAction: TextInputAction.next,
                                          keyboardType:
                                              TextInputType.emailAddress,
                                          prefix: const Icon(Icons.person),
                                          onSave: (email) {
                                            _formData['email'] = email ?? "";
                                          },
                                          validate: (email) {
                                            if (email!.isEmpty ||
                                                email.length < 8 ||
                                                !email.contains("@") ||
                                                email.contains(" ")) {
                                              return "Invalid email address";
                                            } else {
                                              return null;
                                            }
                                          },
                                        ),
                                      ),

                                      Padding(
                                        padding: const EdgeInsets.all(8.0),
                                        child: LoginTextField(
                                          hintText: "Password",
                                          keyboardType: TextInputType.visiblePassword,
                                          prefix: const Icon(Icons.fingerprint),
                                          onSave: (password) {
                                            _formData['password'] =
                                                password ?? "";
                                          },
                                          validate: (password) {
                                            if (password!.isEmpty ||
                                                password.length < 8 ||
                                                password.contains(" ")) {
                                              return "Incorrect password";
                                            } else {
                                              return null;
                                            }
                                          },
                                          isPassword: isPasswordHide,
                                          suffix: IconButton(
                                            onPressed: () {
                                              setState(() {
                                                isPasswordHide = !isPasswordHide;
                                              });
                                            },
                                            icon: isPasswordHide
                                                ? const Icon(Icons.visibility_off)
                                                : const Icon(Icons.visibility),
                                          ),
                                        ),
                                      ),

                                      // forgot password button
                                      Padding(
                                        padding: const EdgeInsets.only(
                                            top: 5,
                                            bottom: 10,
                                            left: 15,
                                            right: 15),
                                        child: Align(
                                          alignment: Alignment.bottomRight,
                                          child: GestureDetector(
                                            onTap: () {
                                              Navigator.push(context, CustomPageRouteDirection(child: const ForgotPasswordPage(), direction: AxisDirection.down));
                                            },
                                            child: const Text(
                                              "Forgot Password?",
                                              style: TextStyle(
                                                  color: Colors.grey,
                                                  fontWeight: FontWeight.bold),
                                              textAlign: TextAlign.right,
                                            ),
                                          ),
                                        ),
                                      ),

                                      // login button
                                      Padding(
                                        padding: const EdgeInsets.all(8.0),
                                        child: PrimaryButton(
                                          title: "Login",
                                          onPressed: () {
                                            if (_formKey.currentState!
                                                .validate()) {
                                              _onSubmit();
                                            }
                                          },
                                        ),
                                      ),
                                    ],
                                  ),
                                ),

                                // create new account button
                                Padding(
                                  padding: const EdgeInsets.only(top: 50),
                                  child: Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    crossAxisAlignment: CrossAxisAlignment.center,
                                    children: [
                                      const Text(
                                        "Don't have an account?",
                                      ),
                                      SecondaryButton(
                                          title: "Create new account",
                                          onPressed: () {
                                            Navigator.push(
                                                context,
                                                CustomPageRoute(child: const RegisterUser())
                                            );
                                          }),
                                    ],
                                  ),
                                )
                              ],
                            ),
                          ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
