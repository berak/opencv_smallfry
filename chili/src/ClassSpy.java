import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Member;
import static java.lang.System.out;

enum ClassMember { CONSTRUCTOR, FIELD, METHOD, CLASS, ALL }

public class ClassSpy {
	// ClassSpy.reflect("org.opencv.core.Mat","ALL");
    public static void reflect(String... args) {
		try {
		    Class<?> c = Class.forName(args[0]);
		    out.format("Class:%n  %s%n%n", c.getCanonicalName());

		    Package p = c.getPackage();
		    String pname = ( p!=null) ? p.getName() + "." : "";
		    out.format("Package:%n  %s%n%n",(p != null ? p.getName() : "-- No Package --"));

		    for (int i = 1; i < args.length; i++) {
				switch (ClassMember.valueOf(args[i])) {
				case CONSTRUCTOR:
				    printMembers(c.getConstructors(), "Constructor", pname);
				    break;
				case FIELD:
				    printMembers(c.getFields(), "Fields", pname);
				    break;
				case METHOD:
				    printMembers(c.getMethods(), "Methods", pname);
				    break;
				case CLASS:
				    printClasses(c);
				    break;
				case ALL:
				    printMembers(c.getConstructors(), "Constuctors", pname);
				    printMembers(c.getFields(), "Fields", pname);
				    printMembers(c.getMethods(), "Methods", pname);
				    printClasses(c);
				    break;
				default:
				    assert false;
				}
		    }
		} catch (ClassNotFoundException x) {
		    x.printStackTrace();
		}
    }

    private static void printMembers(Member[] mbrs, String s, String pname) {
		out.format("%s:%n", s);
		for (Member mbr : mbrs) {
		    if (mbr instanceof Field)
			out.format("  %s%n", ((Field)mbr).toGenericString().replace(pname,""));
		    else if (mbr instanceof Constructor)
			out.format("  %s%n", ((Constructor)mbr).toGenericString().replace(pname,""));
		    else if (mbr instanceof Method)
			out.format("  %s%n", ((Method)mbr).toGenericString().replace(pname,""));
		}
		if (mbrs.length == 0)
		    out.format("  -- No %s --%n", s);
		out.format("%n");
    }

    private static void printClasses(Class<?> c) {
		out.format("Classes:%n");
		Class<?>[] clss = c.getClasses();
		for (Class<?> cls : clss)
		    out.format("  %s%n", cls.getCanonicalName());
		if (clss.length == 0)
		    out.format("  -- No member interfaces, classes, or enums --%n");
		out.format("%n");
    }
}
